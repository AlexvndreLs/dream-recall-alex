"""Recompute du panneau PSD (colonne gauche, Fig. 3 chap.1 d'Arthur).

Produit la courbe PSD continue (spectre Welch complet 1-45Hz) moyennee sur les 19
electrodes et sur les sujets, separement pour High Recallers (HR) et Low Recallers
(LR), en S2. C'est le panneau gauche de la Fig. 3 (Fig. 1A dans la these : "PSD
averaged across all electrodes in S2", courbe LR bleue vs HR rouge).

Pourquoi un recompute
---------------------
feat_extract_umap_fooof_v4.py calcule bien le spectre complet (compute_psd_spectrum)
mais ne sauvegarde QUE les features derivees par bande (psd_{band}, ...). Le spectre
continu n'existe nulle part sur disque -> on le recalcule ici depuis les epochs
preprocessees, en REUTILISANT les fonctions du pipeline (aucune reecriture de la
logique Welch : meme fenetre Hann, meme WINDOW, meme SF), pour garantir la coherence
stricte avec les features deja extraites.

Reutilise depuis feat_extract_umap_fooof_v4.py (source unique de verite) :
  - load_epochs_by_atomic_stage(deriv_path, sub_id) : (n_epochs,19,N_SAMPLES) par stade
  - compute_psd_spectrum(data)                       : (n_epochs,19,n_freqs), freqs

Agregation (fidele Fig.1A / Fig.3 gauche)
-----------------------------------------
Par sujet : moyenne du spectre sur ses epochs S2, puis sur les 19 electrodes
-> 1 spectre par sujet. Puis moyenne sur sujets par groupe (HR, LR). On stocke aussi
l'erreur standard inter-sujets (SEM) pour un ruban d'incertitude optionnel au plot.

Entrees : {deriv_path}/sub-XX/eeg/sub-XX_task-sleep_proc-clean_eeg.vhdr (+ events.tsv)
Sorties : {out_dir}/fig3_psd_spectrum_{state}.npz
          freqs (n_freqs,), psd_hr/psd_lr (n_freqs,), sem_hr/sem_lr (n_freqs,),
          per_subject (n_sujets, n_freqs), labels (n_sujets,)

Ne fait AUCUN plot (separation calcul/visu). Le plot consommera le .npz.

Usage
-----
    python recompute_psd_spectrum_fig3.py \\
        --deriv-path /scratch/alouis/dream_bids/derivatives/preprocessed-noica \\
        --out-dir    /scratch/alouis/dream_features_noica_1000hz_corrected \\
        --state      S2 \\
        --n-jobs     $SLURM_CPUS_PER_TASK

Author: recompute pour Alex (replique Arthur chap.1, panneau PSD)
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed

from config_v3 import (
    SUBJECT_LIST_ORDERED,
    SUBJECT_LABELS,
    CLASSIFICATION_GROUPS,
)
# Reutilisation directe des primitives du pipeline (coherence Welch/Hann/WINDOW).
from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage,
    compute_psd_spectrum,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Dossier BIDS derivatives preprocessed (proc-clean).")
    p.add_argument("--out-dir",    type=Path, required=True)
    p.add_argument("--state",      type=str, default="S2")
    p.add_argument("--n-jobs",     type=int, default=1)
    p.add_argument("--overwrite",  action="store_true", default=False)
    return p.parse_args()


def subject_mean_spectrum(deriv_path, sub_id, state):
    """Spectre moyen d'UN sujet en 'state' : moyenne sur epochs puis sur 19 elec.

    Retourne (freqs, spectrum(n_freqs,)) ou None si le sujet n'a pas d'epochs.
    Concatene les stades atomiques du groupe de classification (ex SWS=S3+S4).
    """
    atomic = CLASSIFICATION_GROUPS[state]
    by_stage = load_epochs_by_atomic_stage(deriv_path, sub_id)
    parts = [by_stage[s] for s in atomic if s in by_stage]
    if not parts:
        return None
    data = np.concatenate(parts, axis=0)            # (n_epochs, 19, N_SAMPLES)
    psds, freqs = compute_psd_spectrum(data)        # (n_epochs, 19, n_freqs), (n_freqs,)
    # moyenne sur epochs puis sur electrodes -> (n_freqs,)
    spectrum = psds.mean(axis=0).mean(axis=0)
    return freqs, spectrum


def _worker(deriv_path, sub_id, label, state):
    try:
        res = subject_mean_spectrum(deriv_path, sub_id, state)
        if res is None:
            return None
        freqs, spec = res
        return (sub_id, label, freqs, spec)
    except Exception as e:
        print(f"  ERROR sub-{sub_id}: {e}")
        return None


def main():
    args = parse_args()
    t0 = time()

    out = args.out_dir / f"fig3_psd_spectrum_{args.state}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    jobs = list(zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS))
    print(f"[{args.state}] calcul spectre pour {len(jobs)} sujets...")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_worker)(args.deriv_path, f"{sid:02d}", lab, args.state)
        for sid, lab in jobs
    )
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("Aucun sujet charge — verifier deriv-path.")

    # verif coherence de l'axe frequentiel entre sujets
    freqs = results[0][2]
    for _, _, f, _ in results:
        if not np.allclose(f, freqs):
            raise RuntimeError("Axe frequentiel incoherent entre sujets.")

    labels = np.array([r[1] for r in results])
    per_subject = np.array([r[3] for r in results])   # (n_sujets, n_freqs)

    def grp(lab):
        sel = per_subject[labels == lab]
        mean = sel.mean(axis=0)
        sem = sel.std(axis=0, ddof=1) / np.sqrt(len(sel))
        return mean, sem

    psd_lr, sem_lr = grp(0)
    psd_hr, sem_hr = grp(1)
    n_hr = int((labels == 1).sum())
    n_lr = int((labels == 0).sum())
    print(f"  HR={n_hr}, LR={n_lr}, n_freqs={len(freqs)} "
          f"({freqs[0]:.1f}-{freqs[-1]:.1f}Hz)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        freqs=freqs,
        psd_hr=psd_hr, psd_lr=psd_lr,
        sem_hr=sem_hr, sem_lr=sem_lr,
        per_subject=per_subject,
        labels=labels,
        n_hr=n_hr, n_lr=n_lr,
        state=args.state,
    )
    print(f"Sauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()