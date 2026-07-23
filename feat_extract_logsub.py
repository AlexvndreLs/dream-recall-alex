"""Extraction psd_logsub_* : oscillatoire par SOUSTRACTION LOG (definition specparam).

Troisieme et derniere formulation du residu oscillatoire :

    psd_osc_{b}     P / A                  ratio lineaire      (actuel)
    psd_sub_{b}     P - A                  soustraction lineaire
    psd_logsub_{b}  log10(P) - log10(A)    soustraction log     <- ce script

ou A = 10 ** ap_fit_log est le fit aperiodic FOOOF reconstruit en lineaire.

Pourquoi ce bras
----------------
La soustraction log est la formulation CANONIQUE de specparam : le modele est
pose en log10-power (log10(PSD) ~ aperiodic_fit + somme de gaussiennes), le
spectre aplati interne est obtenu par soustraction du fit en log, et les
hauteurs de pic (PW de get_params("peak")) sont des hauteurs au-dessus du fit
aperiodic en unites log10. Aucune division n'intervient dans specparam.

Cette definition avait ete implementee le 23/06 puis retiree au profit du ratio
lineaire, pour deux raisons internes au pipeline : garder psd_osc_* dans le meme
espace que psd_* (comparabilite LDA), et parce que band_power fait une moyenne
arithmetique, definie en puissance lineaire. Le critere "fidelite a specparam"
n'avait pas ete pose. Ce bras comble ce manque.

Attention, ratio et soustraction log ne sont PAS equivalents ici
----------------------------------------------------------------
Spectre par spectre, ratio = 10 ** (soustraction log), donc meme information.
Mais band_power moyenne sur les bins de la bande AVANT classification, et
mean(log x) != log(mean x) : la moyenne geometrique implicite du log pondere
differemment les bins que la moyenne arithmetique du ratio. Les accuracies
peuvent donc differer, et le LDA n'est invariant que par transformation affine,
pas monotone. Resultat non predictible a priori.

Valeurs attendues : centrees autour de 0, signees (negatives quand la PSD passe
sous le fit 1/f). Ni band_power ni LDA n'en sont genes.

Sortie
------
save_path/psd_logsub_{band}/psd_logsub_{band}_s{sub}_{stage}.npz  (n_epochs, 19)
plus un symlink cov/ -> <branche overlap>/cov/, requis par
compute_global_n_trials (REF_KEY="cov") et garantissant le meme n_trials_min
que les branches overlap et sub, donc des accuracies comparables.

Usage
-----
    python feat_extract_logsub.py \
        --deriv-path /scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica \
        --save-path  /scratch/alouis/dream_features_noica_1000hz_logsub \
        --cov-source /scratch/alouis/dream_features_noica_1000hz_overlap \
        --n-jobs     4
"""

import argparse
import traceback
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed
from specparam import SpectralGroupModel

from config_v3 import FREQ_DICT, FOOOF_FREQ_RANGE, SUBJECT_IDS
from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage,
    compute_psd_spectrum,
    band_power,
    _vhdr,
)

LOGSUB_KEYS = [f"psd_logsub_{b}" for b in FREQ_DICT]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True)
    p.add_argument("--save-path",  type=Path, required=True)
    p.add_argument("--cov-source", type=Path, required=True,
                   help="Branche fournissant cov/ (symlink). Doit etre la meme "
                        "que celle des psd_osc_* pour que n_trials_min soit "
                        "identique et les accuracies comparables.")
    p.add_argument("--n-jobs",     type=int, default=4)
    p.add_argument("--overwrite",  action="store_true", default=False)
    return p.parse_args()


def fit_fooof_logsub(psds: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Residu oscillatoire par soustraction en espace log10.

    Identique a fit_fooof (feat_extract_umap_fooof_v4) et fit_fooof_sub
    (feat_extract_sub) jusqu'au calcul de ap_fit_log, puis soustraction en log
    au lieu de division ou de soustraction lineaire.

    ap_fit_log EST deja en log10 (offsets - exponents * log10(freqs)), donc on
    soustrait directement de log10(P) sans repasser par 10**.

    Returns : (n_epochs, n_ch, n_freqs), sans dimension, signe.
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs)

    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)

    aperiodic  = fg.get_params("aperiodic")
    offsets    = aperiodic[:, 0:1]
    exponents  = aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]

    # Welch peut retourner des zeros exacts sur certains bins (bande coupee par
    # le filtre, canal plat). log10(0) = -inf contaminerait la moyenne de bande
    # entiere. On plancherait sur le plus petit positif du tableau plutot que
    # sur une constante arbitraire, pour rester a l'echelle des donnees.
    pos = flat_psds[flat_psds > 0]
    if pos.size == 0:
        raise RuntimeError("PSD entierement nulle, donnees corrompues.")
    floor = pos.min()
    n_zero = int((flat_psds <= 0).sum())
    if n_zero:
        print(f"    {n_zero} bins PSD <= 0 planchees a {floor:.3e}", flush=True)
    safe = np.maximum(flat_psds, floor)

    flat_log = np.log10(safe) - ap_fit_log
    return flat_log.reshape(n_epochs, n_ch, n_freqs)


def process_subject(deriv_path: Path, save_path: Path, sub_id: str,
                    overwrite: bool = False) -> None:
    if not _vhdr(deriv_path, sub_id).exists():
        print(f"sub-{sub_id}: derivative absent, skip", flush=True)
        return

    try:
        atomic_epochs = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        print(f"sub-{sub_id}: ERROR loading\n{traceback.format_exc()}", flush=True)
        return

    for stage, data in atomic_epochs.items():
        print(f"  sub-{sub_id} {stage}: {data.shape[0]} epochs", flush=True)

        if not overwrite and all(
            (save_path / k / f"{k}_s{sub_id}_{stage}.npz").exists()
            for k in LOGSUB_KEYS
        ):
            print(f"  sub-{sub_id} {stage}: deja cache, skip", flush=True)
            continue

        try:
            psds, freqs = compute_psd_spectrum(data)
            flat_log = fit_fooof_logsub(psds, freqs)
        except Exception:
            print(f"sub-{sub_id} {stage}: ERROR features\n"
                  f"{traceback.format_exc()}", flush=True)
            continue

        for band, (fmin, fmax) in FREQ_DICT.items():
            key = f"psd_logsub_{band}"
            arr = band_power(flat_log, freqs, fmin, fmax)   # (n_epochs, 19)
            out = save_path / key / f"{key}_s{sub_id}_{stage}.npz"
            if not out.exists() or overwrite:
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=arr)

    print(f"sub-{sub_id}: done", flush=True)


def link_cov(save_path: Path, cov_source: Path) -> None:
    src = cov_source / "cov"
    dst = save_path / "cov"
    if not src.exists():
        raise RuntimeError(f"cov source absent : {src}")
    if dst.exists() or dst.is_symlink():
        print(f"cov : {dst} existe deja, inchange", flush=True)
        return
    save_path.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=True)
    n = len(list(src.glob("*.npz")))
    print(f"cov : symlink {dst} -> {src.resolve()}  ({n} fichiers)", flush=True)


if __name__ == "__main__":
    args = parse_args()
    t0 = time()

    print("=== symlink cov (n_trials_min de reference) ===", flush=True)
    link_cov(args.save_path, args.cov_source)

    print("\n=== extraction psd_logsub_* (soustraction log) ===", flush=True)
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(args.deriv_path, args.save_path,
                                 sub_id, args.overwrite)
        for sub_id in SUBJECT_IDS
    )

    print("\n=== recap ===", flush=True)
    for key in LOGSUB_KEYS:
        d = args.save_path / key
        n = len(list(d.glob("*.npz"))) if d.exists() else 0
        print(f"  {key:20s} : {n} fichiers", flush=True)

    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal: {m}m{s:02d}s", flush=True)
