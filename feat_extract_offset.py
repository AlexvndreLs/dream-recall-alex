"""Extraction slim de l'OFFSET aperiodique FOOOF (parametre b du fond 1/f).

Pourquoi ce script existe
-------------------------
feat_extract_umap_fooof_v4.py calcule bien les deux parametres du fond
aperiodique (fit_fooof, ligne 234 : fg.get_params("aperiodic") renvoie
(n_spectres, 2), col0 = offset, col1 = exposant) mais ne sauvegarde QUE
l'exposant, sous la key "aperiodic". La colonne offset ne sert qu'a
reconstruire ap_fit_log en interne, puis est jetee. Idem dans
feat_extract_sub.py et feat_extract_logsub.py. L'offset n'existe donc nulle
part sur disque (verifie par find sur /scratch/alouis le 02/09/2026).

Ce script refait le strict minimum pour le recuperer : Welch + fit FOOOF.
Pas de covariance, pas de cospectres, pas de complexite, pas de puissance
par bande. Il ecrit une nouvelle feature vectorielle "aperiodic_offset",
de meme forme que "aperiodic" : (n_epochs, 19), un fichier par sujet et par
stade atomique.

Point critique : le parametre d'overlap Welch
---------------------------------------------
config_v3.py contient aujourd'hui OVERLAP = 500, valeur utilisee pour les
branches *_overlap. La branche de replication stricte
dream_features_noica_1000hz a ete produite AVANT, avec un overlap nul.
Importer OVERLAP depuis la config produirait donc un offset incomparable a
l'exposant deja sur disque dans cette branche.

L'overlap est pour cette raison un argument CLI explicite, defaut 0, et le
mode --verify recalcule l'exposant au passage puis le compare a celui deja
stocke. Si l'overlap est faux, l'ecart saute aux yeux. Ne pas desactiver
--verify sur une premiere execution.

Usage
-----
    # branche de replication stricte (no-overlap) : overlap 0
    python feat_extract_offset.py \
        --deriv-path /home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica \
        --save-path  /scratch/alouis/dream_features_noica_1000hz \
        --welch-overlap 0 \
        --n-jobs $SLURM_CPUS_PER_TASK \
        --verify

    # branche overlap : overlap 500
    python feat_extract_offset.py \
        --deriv-path /home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica \
        --save-path  /scratch/alouis/dream_features_noica_1000hz_overlap \
        --welch-overlap 500 \
        --n-jobs $SLURM_CPUS_PER_TASK \
        --verify

Sortie
------
    {save-path}/aperiodic_offset/aperiodic_offset_s{XX}_{stage}.npz  (cle "data")

Ensuite
-------
    python classify.py --save-path {save-path} --key aperiodic_offset \
        --n-perm 1000 --n-bootstraps 1000 --n-jobs $SLURM_CPUS_PER_TASK
    python compute_maxstat_correction.py --save-path {save-path} \
        --output-path {save-path}_corrected --family-name unused \
        --mode arthur --keys aperiodic_offset
"""

import argparse
import os
import traceback
from pathlib import Path
from time import time

import numpy as np
import mne
from joblib import Parallel, delayed
from specparam import SpectralGroupModel

from config_v3 import (
    SFREQ_PREPROC,
    WINDOW,
    FOOOF_FREQ_RANGE,
    SUBJECT_IDS,
)
# Reutilise le decoupage en epochs du script principal plutot que de le
# reecrire : garantit un alignement exact epoch par epoch avec les .npz
# "aperiodic" deja presents (meme scorer, meme rejet des blocs incomplets,
# meme ordre).
from feat_extract_umap_fooof_v4 import load_epochs_by_atomic_stage, _vhdr
from utils import load_atomic

SF = int(SFREQ_PREPROC)
KEY = "aperiodic_offset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine du derivative preprocessed (ex: .../preprocessed-noica)")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Branche de features cible, celle qui contient deja aperiodic/")
    p.add_argument("--welch-overlap", type=int, required=True,
                   help="n_overlap Welch en samples. 0 pour la branche de replication "
                        "stricte, 500 pour les branches _overlap. NE PAS deviner : "
                        "verifier avec --verify.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--verify", action="store_true", default=False,
                   help="Recalcule aussi l'exposant et le compare a celui deja "
                        "stocke sous la key 'aperiodic'. Controle que l'overlap "
                        "et les parametres Welch sont les bons.")
    p.add_argument("--verify-tol", type=float, default=1e-6,
                   help="Ecart absolu max tolere sur l'exposant recalcule.")
    return p.parse_args()


def save_atomic(out: Path, arr: np.ndarray) -> None:
    """Ecrit via un temporaire puis os.replace.

    np.savez_compressed ecrit directement a destination : un job tue en plein
    write (timeout SLURM, OOM) laisserait un .npz tronque que la reprise
    considererait comme deja fait. os.replace est atomique sur un meme
    systeme de fichiers, donc le fichier final n'apparait que complet.
    N'ecrase jamais un fichier existant : l'appelant verifie avant.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    try:
        # np.savez_compressed AJOUTE ".npz" si on lui passe un chemin qui ne
        # finit pas par ".npz". Passer un descripteur ouvert desactive ce
        # comportement et garantit que le fichier ecrit porte bien le nom tmp.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, data=arr)
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink()   # ne supprime que le temporaire cree juste au-dessus
        raise


def psd_welch(data: np.ndarray, n_overlap: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_epochs, 19, n_samples) -> psds (n_epochs, 19, n_freqs), freqs.

    Parametres identiques a compute_psd_spectrum de feat_extract_umap_fooof_v4,
    a ceci pres que n_overlap est passe explicitement au lieu d'etre importe
    de la config.
    """
    return mne.time_frequency.psd_array_welch(
        data,
        sfreq=SF,
        fmin=FOOOF_FREQ_RANGE[0],
        fmax=FOOOF_FREQ_RANGE[1],
        n_fft=WINDOW,
        n_overlap=n_overlap,
        n_per_seg=WINDOW,
        window="hann",
        verbose=False,
    )


def fit_aperiodic(psds: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit FOOOF mode fixed -> (offset, exponent), chacun (n_epochs, 19).

    Meme appel que fit_fooof du script principal : SpectralGroupModel avec
    aperiodic_mode="fixed", n_jobs=1 puisque le parallelisme est au niveau
    sujet. get_params("aperiodic") renvoie (n_spectres, 2), col0 = offset,
    col1 = exposant.
    """
    n_epochs, n_ch, n_freqs = psds.shape
    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, psds.reshape(-1, n_freqs), freq_range=FOOOF_FREQ_RANGE, n_jobs=1)
    ap = fg.get_params("aperiodic")
    offset = ap[:, 0].reshape(n_epochs, n_ch)
    exponent = ap[:, 1].reshape(n_epochs, n_ch)
    return offset, exponent


def process_subject(deriv_path: Path, save_path: Path, sub_id: str,
                    n_overlap: int, overwrite: bool, verify: bool) -> list[str]:
    """Retourne une liste de lignes de rapport (verification ou erreurs)."""
    report: list[str] = []

    if not _vhdr(deriv_path, sub_id).exists():
        report.append(f"sub-{sub_id}: derivative absent, skip")
        return report

    try:
        atomic_epochs = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        report.append(f"sub-{sub_id}: ERREUR chargement\n{traceback.format_exc()}")
        return report

    for stage, data in atomic_epochs.items():
        out = save_path / KEY / f"{KEY}_s{sub_id}_{stage}.npz"
        if out.exists() and not overwrite and not verify:
            continue

        try:
            psds, freqs = psd_welch(data, n_overlap)
            offset, exponent = fit_aperiodic(psds, freqs)
        except Exception:
            report.append(f"sub-{sub_id} {stage}: ERREUR fit\n{traceback.format_exc()}")
            continue

        if verify:
            ref = load_atomic(save_path, "aperiodic", sub_id, stage)
            if ref is None:
                report.append(f"sub-{sub_id} {stage}: VERIF impossible, "
                              f"aperiodic_s{sub_id}_{stage}.npz absent")
            elif ref.shape != exponent.shape:
                report.append(f"sub-{sub_id} {stage}: VERIF forme differente, "
                              f"stocke {ref.shape} vs recalcule {exponent.shape}")
            else:
                dmax = float(np.abs(ref - exponent).max())
                report.append(f"sub-{sub_id} {stage}: verif exposant, "
                              f"ecart max {dmax:.3e} sur {exponent.shape[0]} epochs")

        if not out.exists() or overwrite:
            save_atomic(out, offset)

    return report


if __name__ == "__main__":
    args = parse_args()
    t0 = time()

    print("=== extraction offset aperiodique ===")
    print(f"deriv        : {args.deriv_path}")
    print(f"save         : {args.save_path}")
    print(f"welch overlap: {args.welch_overlap} samples (fenetre {WINDOW}, hann)")
    print(f"verify       : {args.verify}")
    print()

    reports = Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(
            args.deriv_path, args.save_path, sub_id,
            args.welch_overlap, args.overwrite, args.verify,
        )
        for sub_id in SUBJECT_IDS
    )

    lines = [ln for rep in reports for ln in rep]
    for ln in lines:
        print(ln)

    if args.verify:
        diffs = []
        for ln in lines:
            if "ecart max" in ln:
                diffs.append(float(ln.split("ecart max")[1].split("sur")[0]))
        if diffs:
            worst = max(diffs)
            print()
            print(f"VERIFICATION : {len(diffs)} couples sujet x stade compares, "
                  f"ecart absolu max sur l'exposant = {worst:.3e}")
            if worst > args.verify_tol:
                print(f"  ECHEC : au-dela de la tolerance {args.verify_tol:.1e}. "
                      f"Les parametres Welch ne correspondent pas a ceux ayant "
                      f"produit la branche. Verifier --welch-overlap avant "
                      f"d'utiliser les offsets produits.")
            else:
                print(f"  OK : sous la tolerance {args.verify_tol:.1e}. Les offsets "
                      f"sont comparables a l'exposant deja stocke.")

    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal: {m}m{s:02d}s")
