"""Compare Riemannian Potato Field (rejet unifie) vs potato par-feature
(deja calcule dans dream_features_noica_1000hz_overlap_potato/) sur tous
les sujets et tous les stades atomiques (S1, S2, S3, S4, REM).

Ne filtre rien de nouveau sur disque -- calcule juste les % de retention
pour comparer les deux approches et estimer l'impact sur n_trials_min.

Usage:
    python compare_rpf_vs_perfeature.py \
        --raw-dir    /home/alouis/scratch/dream_features_noica_1000hz_overlap \
        --potato-dir /home/alouis/scratch/dream_features_noica_1000hz_overlap_potato \
        --out-csv    /home/alouis/scratch/rpf_vs_perfeature_comparison.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.clustering import PotatoField

KEYS = ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma", "cosp_beta"]
ATOMIC_STAGES = ["S1", "S2", "S3", "S4", "REM"]
SUBJECTS = [f"{i:02d}" for i in range(1, 39)]  # s01..s38 -- ajuste si numerotation differente


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--potato-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--p-threshold", type=float, default=0.01)
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--n-iter-max", type=int, default=300)
    return p.parse_args()


def load_key(base_dir: Path, key: str, sub_id: str, stage: str):
    f = base_dir / key / f"{key}_s{sub_id}_{stage}.npz"
    if not f.exists():
        return None
    return np.load(f)["data"]


def main():
    args = parse_args()
    rows = []

    for sub_id in SUBJECTS:
        for stage in ATOMIC_STAGES:
            # --- charger les 6 features brutes pour ce sujet/stade ---
            X_raw = []
            missing = False
            for key in KEYS:
                arr = load_key(args.raw_dir, key, sub_id, stage)
                if arr is None:
                    missing = True
                    break
                X_raw.append(arr)
            if missing:
                print(f"SKIP sub-{sub_id}/{stage} : fichier brut manquant")
                continue

            n_epochs = [x.shape[0] for x in X_raw]
            if len(set(n_epochs)) != 1:
                print(f"SKIP sub-{sub_id}/{stage} : n_epochs incoherents avant filtrage {dict(zip(KEYS, n_epochs))}")
                continue
            n_before = n_epochs[0]

            if n_before < 10:
                # meme garde-fou que apply_potato_filter.py : trop peu d'epochs pour fitter
                rows.append(dict(sub_id=sub_id, stage=stage, n_before=n_before,
                                  n_after_rpf=n_before, pct_rpf=100.0,
                                  pct_perfeature_mean=100.0, pct_perfeature_min=100.0,
                                  skipped_too_few=True))
                continue

            # --- RPF : une seule decision par epoch ---
            t0 = time.time()
            pf = PotatoField(n_potatoes=len(KEYS), p_threshold=args.p_threshold,
                              z_threshold=args.z_threshold, n_iter_max=args.n_iter_max)
            pf.fit(X_raw)
            mask = pf.predict(X_raw)
            n_after_rpf = int(mask.sum())
            pct_rpf = 100 * n_after_rpf / n_before
            elapsed = time.time() - t0

            # --- potato par-feature : deja calcule, juste compter les fichiers existants ---
            pct_per_key = {}
            for key in KEYS:
                arr_potato = load_key(args.potato_dir, key, sub_id, stage)
                if arr_potato is not None:
                    pct_per_key[key] = 100 * len(arr_potato) / n_before

            pct_mean = np.mean(list(pct_per_key.values())) if pct_per_key else float("nan")
            pct_min = np.min(list(pct_per_key.values())) if pct_per_key else float("nan")

            rows.append(dict(
                sub_id=sub_id, stage=stage, n_before=n_before,
                n_after_rpf=n_after_rpf, pct_rpf=round(pct_rpf, 1),
                pct_perfeature_mean=round(pct_mean, 1) if not np.isnan(pct_mean) else None,
                pct_perfeature_min=round(pct_min, 1) if not np.isnan(pct_min) else None,
                skipped_too_few=False, fit_seconds=round(elapsed, 1),
            ))
            print(f"sub-{sub_id}/{stage}: n={n_before} RPF={pct_rpf:.1f}% "
                  f"per-feature(min/mean)={pct_min:.1f}%/{pct_mean:.1f}% ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print()
    print("=== RESUME GLOBAL ===")
    valid = df[~df["skipped_too_few"]]
    print(f"RPF        : retention moyenne = {valid['pct_rpf'].mean():.1f}%  "
          f"min = {valid['pct_rpf'].min():.1f}%")
    print(f"Per-feature: retention moyenne = {valid['pct_perfeature_mean'].mean():.1f}%  "
          f"min (pire feature/sujet) = {valid['pct_perfeature_min'].min():.1f}%")
    print()
    print(f"n_after_rpf min (pire sujet/stade) = {valid['n_after_rpf'].min()}  "
          f"-> impact potentiel sur n_trials_min si on adopte RPF")
    print(f"CSV : {args.out_csv}")


if __name__ == "__main__":
    main()
