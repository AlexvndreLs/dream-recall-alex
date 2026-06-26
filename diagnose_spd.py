"""Diagnostic SPD des features matricielles (cov, cosp_*) après average reference.

Pourquoi : l'average reference (preprocess_subject_v3.py, étape 5) projette les
19 canaux dans un espace de rang 18 (les canaux somment à 0). Les matrices de
covariance et de cospectre deviennent alors singulières (rang 18, plus petite
valeur propre ~ 0). Le log-map Riemannien du TSclassifier (classify.py) exige des
matrices SPD STRICTES et produit des NaN sur une matrice singulière.

Ce script lit les .npz réels produits par feat_extract et rapporte, pour chaque
feature matricielle et chaque stade, la plus petite valeur propre, le rang et le
conditionnement. Il vérifie ensuite que le shrinkage SPD_SHRINK de classify.py
(SPDRegularizer) suffit à restaurer le SPD strict sur TES données.

À lancer sur le cluster APRÈS feat_extract, AVANT classify :

    cd ~/dream-recall-alex && source /home/alouis/mne_env/bin/activate
    python diagnose_spd.py --save-path /home/alouis/scratch/dream_features

Interprétation :
  - "min_eig brut < 1e-10" sur une ligne  -> CONFIRME la singularité (attendu après CAR)
  - "min_eig shrink > 1e-6" partout        -> le patch classify.py suffit (OK pour lancer)
  - "min_eig shrink encore < 1e-6"         -> augmenter SPD_SHRINK dans classify.py
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.linalg import eigvalsh, matrix_rank, cond

from config_v3 import (
    FREQ_DICT, CLASSIFICATION_GROUPS, STATE_LIST,
    SUBJECT_LIST_ORDERED,
)
from utils import load_atomic

try:
    from classify import SPD_SHRINK, SPDRegularizer
except Exception:
    SPD_SHRINK = 1e-3
    SPDRegularizer = None


MATRIX_KEYS = ["cov"] + [f"cosp_{b}" for b in FREQ_DICT]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--max-subjects", type=int, default=6,
                   help="Nombre de sujets échantillonnés par (feature, stade).")
    p.add_argument("--shrink", type=float, default=SPD_SHRINK)
    return p.parse_args()


def load_state(save_path, key, sub_id, state):
    stages = CLASSIFICATION_GROUPS[state]
    parts = [a for s in stages if (a := load_atomic(save_path, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None


def regularize(mats, shrink):
    n = mats.shape[-1]
    mu = np.trace(mats, axis1=-2, axis2=-1) / n
    return (1 - shrink) * mats + shrink * mu[:, None, None] * np.eye(n)


def summarize(mats):
    """Retourne (min_eig, rang médian, cond max) sur un échantillon de matrices."""
    eigs = np.array([eigvalsh(m).min() for m in mats])
    rks  = np.array([matrix_rank(m) for m in mats])
    conds = []
    for m in mats:
        try:
            conds.append(cond(m))
        except Exception:
            conds.append(np.inf)
    return eigs.min(), int(np.median(rks)), max(conds)


def main():
    args = parse_args()
    print(f"=== Diagnostic SPD  (shrink={args.shrink:.0e}) ===\n")
    print(f"{'feature':14s} {'state':6s} | {'min_eig brut':>13s} {'rang':>5s} {'cond brut':>11s}"
          f" | {'min_eig shrink':>15s} {'cond shrink':>12s}")
    print("-" * 95)

    any_singular = False
    all_fixed = True

    for key in MATRIX_KEYS:
        for state in STATE_LIST:
            sample = []
            for sub_id in SUBJECT_LIST_ORDERED[: args.max_subjects]:
                arr = load_state(args.save_path, key, sub_id, state)
                if arr is not None and len(arr):
                    # échantillonne jusqu'à 20 matrices par sujet pour limiter le coût
                    idx = np.linspace(0, len(arr) - 1, min(20, len(arr))).astype(int)
                    sample.append(arr[idx])
            if not sample:
                continue
            mats = np.concatenate(sample, axis=0)

            mn_raw, rk, cd_raw = summarize(mats)
            mats_r = regularize(mats, args.shrink)
            mn_fix, _, cd_fix = summarize(mats_r)

            n_ch = mats.shape[-1]
            singular = mn_raw < 1e-10 or rk < n_ch
            fixed = mn_fix > 1e-6
            any_singular |= singular
            all_fixed &= fixed

            flag_raw = " <-- SINGULIER" if singular else ""
            flag_fix = "" if fixed else " <-- ENCORE SINGULIER"
            print(f"{key:14s} {state:6s} | {mn_raw:13.3e} {rk:3d}/{n_ch} {cd_raw:11.2e}"
                  f" | {mn_fix:15.3e} {cd_fix:12.2e}{flag_raw}{flag_fix}")

    print("\n=== Conclusion ===")
    if any_singular:
        print("Matrices brutes singulières détectées (attendu après average reference).")
    else:
        print("Aucune singularité brute détectée — surprenant si CAR active. Vérifier le prepro.")
    if all_fixed:
        print(f"Le shrinkage {args.shrink:.0e} restaure le SPD strict partout -> OK pour lancer classify.py.")
    else:
        print(f"Le shrinkage {args.shrink:.0e} INSUFFISANT sur certaines features -> "
              f"augmenter SPD_SHRINK dans classify.py (essayer 1e-2).")


if __name__ == "__main__":
    main()