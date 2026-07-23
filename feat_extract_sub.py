"""Extraction psd_sub_* : oscillatoire par SOUSTRACTION LINEAIRE.

Variante de feat_extract_umap_fooof_v4.py qui remplace le ratio

    flat_ratio = flat_psds / (10 ** ap_fit_log)          # psd_osc_*, sans dim.

par la soustraction en espace lineaire

    flat_sub   = flat_psds - (10 ** ap_fit_log)          # psd_sub_*, V^2/Hz

Motivation
----------
diag (23/07, 4 sujets x 3 stades, log job diag_sub) : les deux definitions ne
sont PAS equivalentes. Correlation ratio vs soustraction par bande :

    sigma  r=0.90-0.93      delta  r=0.39-0.45
    theta  r=0.72-0.90      beta   r=0.72-0.84

et la soustraction produit une majorite de valeurs NEGATIVES en delta
(49-74% selon le stade) contre <2% en alpha/sigma : le fit 1/f domine le
spectre en basse frequence, donc P-A change de signe, alors que P/A reste
positif borne autour de 1. Ce sont deux quantites differentes, pas deux
echelles de la meme.

Or les deux resultats vectoriels significatifs de la track clean sont
psd_osc_delta_SWS et psd_osc_beta_SWS, precisement les bandes ou les
definitions divergent le plus. D'ou ce bras de comparaison.

Pas de rescaling
----------------
Meme diag : relerr float64 vs float128 = 0.000e+00 EXACTEMENT sur les 12
combos, facteur d'annulation |P|/|P-A| ~= 4 (moins d'un bit de mantisse
perdu). La soustraction naive est exacte. Un rescaling x1e12 avant
soustraction INTRODUIT au contraire une erreur d'arrondi de ~1.9e-16 (un
ulp, via les deux multiplications et la division). Unites brutes conservees.
Le LDA solver="svd" est de toute facon invariant par transformation affine
sur une feature univariee (cf --normalize, delta accuracy = 0.0000 sur 51
combos).

Sortie
------
save_path/psd_sub_{band}/psd_sub_{band}_s{sub}_{stage}.npz  (n_epochs, 19)

plus des symlinks save_path/cov/ -> <branche overlap>/cov/, requis par
compute_global_n_trials de classify.py (REF_KEY="cov", RuntimeError sinon).
Les .npz de cov (19x19 par epoch) sont volumineux, d'ou le lien plutot que
la copie.

Usage
-----
    python feat_extract_sub.py \
        --deriv-path /home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica \
        --save-path  /home/alouis/scratch/dream_features_noica_1000hz_sub \
        --cov-source /home/alouis/scratch/dream_features_noica_1000hz_overlap \
        --n-jobs     $SLURM_CPUS_PER_TASK
"""

import argparse
import traceback
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed
from specparam import SpectralGroupModel

from config_v3 import (
    FREQ_DICT, FOOOF_FREQ_RANGE, ATOMIC_STAGES, SUBJECT_IDS,
)
from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage,
    compute_psd_spectrum,
    band_power,
    _vhdr,
)

SUB_KEYS = [f"psd_sub_{b}" for b in FREQ_DICT]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True)
    p.add_argument("--save-path",  type=Path, required=True)
    p.add_argument("--cov-source", type=Path, required=True,
                   help="Branche existante fournissant cov/ (symlink). "
                        "Doit etre la MEME branche que celle qui a produit "
                        "psd_osc_*, sinon la comparaison ne tient pas.")
    p.add_argument("--n-jobs",     type=int, default=1)
    p.add_argument("--overwrite",  action="store_true", default=False)
    return p.parse_args()


def fit_fooof_sub(psds: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Residu oscillatoire par soustraction lineaire.

    Identique a fit_fooof de feat_extract_umap_fooof_v4 jusqu'au calcul de
    ap_fit_log, puis soustraction au lieu de division. L'exposant aperiodic
    n'est pas retourne : il est deja extrait a l'identique par la branche
    principale (cle "aperiodic"), inutile de le recalculer ici.

    Returns : (n_epochs, n_ch, n_freqs) en V^2/Hz, valeurs signees.
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs)

    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)

    aperiodic  = fg.get_params("aperiodic")
    offsets    = aperiodic[:, 0:1]
    exponents  = aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]

    flat_sub = flat_psds - (10 ** ap_fit_log)
    return flat_sub.reshape(n_epochs, n_ch, n_freqs)


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
            for k in SUB_KEYS
        ):
            print(f"  sub-{sub_id} {stage}: deja cache, skip", flush=True)
            continue

        try:
            psds, freqs = compute_psd_spectrum(data)
            flat_sub = fit_fooof_sub(psds, freqs)
        except Exception:
            print(f"sub-{sub_id} {stage}: ERROR features\n"
                  f"{traceback.format_exc()}", flush=True)
            continue

        for band, (fmin, fmax) in FREQ_DICT.items():
            key = f"psd_sub_{band}"
            arr = band_power(flat_sub, freqs, fmin, fmax)   # (n_epochs, 19)
            out = save_path / key / f"{key}_s{sub_id}_{stage}.npz"
            if not out.exists() or overwrite:
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=arr)

    print(f"sub-{sub_id}: done", flush=True)


def link_cov(save_path: Path, cov_source: Path) -> None:
    """Symlink cov/ depuis la branche de reference.

    classify.py:compute_global_n_trials leve RuntimeError si aucun .npz de
    REF_KEY="cov" n'est trouve. Le n_trials_min doit en outre etre IDENTIQUE
    a celui de la branche overlap pour que les accuracies soient comparables
    entre psd_osc_* et psd_sub_* : d'ou le lien vers la meme source plutot
    qu'un recalcul.
    """
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

    print("\n=== extraction psd_sub_* (soustraction lineaire) ===", flush=True)
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(args.deriv_path, args.save_path,
                                 sub_id, args.overwrite)
        for sub_id in SUBJECT_IDS
    )

    print("\n=== recap ===", flush=True)
    for key in SUB_KEYS:
        d = args.save_path / key
        n = len(list(d.glob("*.npz"))) if d.exists() else 0
        print(f"  {key:16s} : {n} fichiers", flush=True)

    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal: {m}m{s:02d}s", flush=True)
