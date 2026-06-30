"""DIAGNOSTIC JETABLE — pas un ajout au pipeline.

Compare deux définitions de flattening FOOOF sur quelques sujets :
  (A) ratio linéaire (code actuel) : psd / 10**ap_fit_log
  (B) soustraction linéaire (proposition en commentaire) : psd - 10**ap_fit_log

Pour chaque sujet/stade/bande, calcule les deux versions de psd_osc_*,
les agrège en feature scalaire par canal (comme band_power), puis :
  1. Corrélation (Pearson) entre les deux features à travers les sujets
     -> si très corrélées, le choix importe peu en pratique.
  2. Accuracy LDA leave-one-subject-out rapide pour CHAQUE version
     -> si l'une bat clairement l'autre, ça tranche empiriquement.

Usage (sur Fir, dans mne_env, depuis le dossier contenant
feat_extract_umap_fooof_v4.py et config_v3.py) :

    python compare_fooof_flattening.py \\
        --deriv-path /path/to/derivatives/preprocessed-ica \\
        --stage REM \\
        --band alpha \\
        --n-subjects 10   # par groupe HR/LR, donc 20 sujets max au total

Ne PAS lancer sur les 38 sujets d'un coup pour une première passe : prends
n-subjects=8-10 par groupe pour avoir une réponse en quelques minutes avant
de généraliser.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneGroupOut
from specparam import SpectralGroupModel

from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage,
    compute_psd_spectrum,
    band_power,
)
from config_v3 import FREQ_DICT, FOOOF_FREQ_RANGE, SUBJECT_IDS

# HR = 1..18, LR = 19,20,23..38 (21,22 exclus, cf mémoire projet)
HR_IDS = [s for s in SUBJECT_IDS if 1 <= int(s) <= 18]
LR_IDS = [s for s in SUBJECT_IDS if int(s) in [19, 20] + list(range(23, 39))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True)
    p.add_argument("--stage", type=str, required=True,
                   help="stade atomique, ex: REM, S2, S3, S4, S1")
    p.add_argument("--band", type=str, required=True,
                   help="clé de FREQ_DICT, ex: alpha, sigma, beta")
    p.add_argument("--n-subjects", type=int, default=8,
                   help="nb de sujets PAR GROUPE (HR et LR), défaut 8")
    return p.parse_args()


def both_flattenings(psds: np.ndarray, freqs: np.ndarray):
    """Réplique fit_fooof mais retourne les deux versions du flattening.

    Pas un refactor du pipeline -> duplication volontaire et isolée,
    pour ne pas toucher feat_extract_umap_fooof_v4.py pour un diagnostic.
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs)

    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)
    aperiodic = fg.get_params("aperiodic")
    offsets, exponents = aperiodic[:, 0:1], aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]
    ap_linear = 10 ** ap_fit_log

    ratio_flat = (flat_psds / ap_linear).reshape(n_epochs, n_ch, n_freqs)
    sub_flat   = (flat_psds - ap_linear).reshape(n_epochs, n_ch, n_freqs)
    return ratio_flat, sub_flat


def subject_feature(deriv_path: Path, sub_id: str, stage: str, fmin: float, fmax: float):
    """-> (feat_ratio, feat_sub) : moyenne (epochs, canaux) -> scalaire par sujet."""
    atomic = load_epochs_by_atomic_stage(deriv_path, sub_id)
    if stage not in atomic:
        return None
    data = atomic[stage]
    psds, freqs = compute_psd_spectrum(data)
    ratio_flat, sub_flat = both_flattenings(psds, freqs)
    feat_ratio = band_power(ratio_flat, freqs, fmin, fmax).mean()
    feat_sub   = band_power(sub_flat,   freqs, fmin, fmax).mean()
    return feat_ratio, feat_sub


def quick_loso_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """LDA leave-one-subject-out, X 1D -> reshape (-1,1). Diagnostic rapide,
    PAS un remplacement de classify.py (pas de bootstrap, pas de permutation)."""
    logo = LeaveOneGroupOut()
    clf = LDA(solver="svd")
    correct = 0
    for train_idx, test_idx in logo.split(X.reshape(-1, 1), y, groups):
        clf.fit(X[train_idx].reshape(-1, 1), y[train_idx])
        pred = clf.predict(X[test_idx].reshape(-1, 1))
        correct += (pred == y[test_idx]).sum()
    return correct / len(y)


def main() -> None:
    args = parse_args()
    fmin, fmax = FREQ_DICT[args.band]

    hr_ids = HR_IDS[: args.n_subjects]
    lr_ids = LR_IDS[: args.n_subjects]

    print(f"=== diag flattening FOOOF : stage={args.stage} band={args.band} "
          f"({fmin}-{fmax}Hz) | {len(hr_ids)} HR + {len(lr_ids)} LR ===")

    ratios, subs, labels, groups = [], [], [], []
    for label, ids in [(1, hr_ids), (0, lr_ids)]:
        for sub_id in ids:
            res = subject_feature(args.deriv_path, sub_id, args.stage, fmin, fmax)
            if res is None:
                print(f"  sub-{sub_id}: stage {args.stage} absent, skip")
                continue
            feat_ratio, feat_sub = res
            ratios.append(feat_ratio)
            subs.append(feat_sub)
            labels.append(label)
            groups.append(sub_id)
            print(f"  sub-{sub_id} (label={label}): ratio={feat_ratio:.4f} "
                  f"sub={feat_sub:.6f}")

    ratios = np.asarray(ratios)
    subs   = np.asarray(subs)
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    r, p = pearsonr(ratios, subs)
    print(f"\nCorrélation (ratio vs soustraction) à travers les sujets : "
          f"r={r:.3f} (p={p:.4f})")
    if r > 0.9:
        print("  -> très corrélées : le choix de définition importe peu ici.")
    else:
        print("  -> peu corrélées : les deux définitions capturent des "
              "informations différentes, le choix compte.")

    acc_ratio = quick_loso_accuracy(ratios, labels, groups)
    acc_sub   = quick_loso_accuracy(subs, labels, groups)
    print(f"\nAccuracy LOSO rapide (diagnostic, PAS classify.py officiel) :")
    print(f"  ratio       : {acc_ratio:.3f}")
    print(f"  soustraction: {acc_sub:.3f}")
    print(f"  (chance = {max(np.mean(labels), 1 - np.mean(labels)):.3f})")


if __name__ == "__main__":
    main()