"""Validation de la redondance des features étendues (vs pipeline Arthur).

Le pipeline ajoute trois familles de features absentes d'Arthur :
  1. exposant aperiodic (FOOOF)
  2. PSD oscillatoire psd_osc_{band} (excès au-dessus du 1/f, FOOOF)
  3. complexité : perm_entropy, higuchi_fd, spec_entropy (antropy)

Un reviewer exigera la preuve que ces ajouts apportent une information NON
redondante. Ce script produit cette preuve, en deux volets :

A. Redondance complexité <-> aperiodic (review §4.1)
   - corrélation de Spearman par (stade, canal) entre chaque feature de
     complexité et l'exposant aperiodic. |rho| > 0.8 => quasi-redondant.
   - spec_entropy est la plus suspecte (même spectre Welch que l'aperiodic).

B. Redondance psd_osc <-> psd brut (review §4.2)
   - corrélation par (stade, canal) entre psd_osc_{band} et psd_{band}.
   - cas clé : sigma en S2 (fuseaux). Si psd_osc_sigma est DÉCORRÉLÉ de
     psd_sigma, la séparation FOOOF capture quelque chose de neuf -> argument
     fort pour l'ajout FOOOF.

Sortie : feature_redundancy.csv + verdict par feature.

Usage :
    python validate_features.py --save-path /home/alouis/scratch/dream_features

Léger (corrélations seulement, pas de classification) -> tourne en quelques min.
La validation par accuracy marginale (aperiodic seul vs +complexité) se fait
ensuite via les résultats de classify.py (comparer les acc_mean dans le CSV).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config_v3 import (
    FREQ_DICT, CLASSIFICATION_GROUPS, STATE_LIST,
    SUBJECT_LIST_ORDERED, N_EEG, CH_NAMES,
)
from utils import load_atomic

BANDS = list(FREQ_DICT)
COMPLEXITY = ["perm_entropy", "higuchi_fd", "spec_entropy"]
REDUNDANT_THRESHOLD = 0.8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    return p.parse_args()


def load_concat(save_path, key, state):
    """Concatène toutes les epochs de tous les sujets pour (key, state) -> (N, 19)."""
    stages = CLASSIFICATION_GROUPS[state]
    arrays = []
    for sub_id in SUBJECT_LIST_ORDERED:
        parts = [a for s in stages
                 if (a := load_atomic(save_path, key, sub_id, s)) is not None]
        if parts:
            arrays.append(np.concatenate(parts, axis=0))
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)  # (N_epochs_total, 19)


def per_channel_spearman(a, b):
    """Corrélation de Spearman canal par canal entre deux arrays (N, 19)."""
    n_ch = a.shape[1]
    out = np.empty(n_ch)
    for c in range(n_ch):
        rho, _ = spearmanr(a[:, c], b[:, c])
        out[c] = rho
    return out


def main():
    args = parse_args()
    rows = []

    print("=== A. Redondance complexité <-> aperiodic ===\n")
    print(f"{'feature':14s} {'state':6s} | {'|rho| moyen':>11s} {'|rho| max':>10s}  verdict")
    print("-" * 70)
    for state in STATE_LIST:
        ap = load_concat(args.save_path, "aperiodic", state)
        if ap is None:
            continue
        for feat in COMPLEXITY:
            arr = load_concat(args.save_path, feat, state)
            if arr is None or arr.shape != ap.shape:
                continue
            rho = per_channel_spearman(arr, ap)
            absrho = np.abs(rho)
            redundant = absrho.mean() > REDUNDANT_THRESHOLD
            verdict = "REDONDANT" if redundant else "info ajoutée"
            print(f"{feat:14s} {state:6s} | {absrho.mean():11.3f} {absrho.max():10.3f}  {verdict}")
            rows.append(dict(comparison="complexity_vs_aperiodic", feature=feat,
                             state=state, abs_rho_mean=float(absrho.mean()),
                             abs_rho_max=float(absrho.max()), redundant=redundant))

    print("\n=== B. Redondance psd_osc <-> psd brut (par bande) ===\n")
    print(f"{'band':8s} {'state':6s} | {'|rho| moyen':>11s} {'|rho| max':>10s}  verdict")
    print("-" * 64)
    for state in STATE_LIST:
        for b in BANDS:
            raw = load_concat(args.save_path, f"psd_{b}", state)
            osc = load_concat(args.save_path, f"psd_osc_{b}", state)
            if raw is None or osc is None or raw.shape != osc.shape:
                continue
            rho = per_channel_spearman(osc, raw)
            absrho = np.abs(rho)
            redundant = absrho.mean() > REDUNDANT_THRESHOLD
            verdict = "REDONDANT" if redundant else "info ajoutée"
            marker = "  <-- cas fuseaux" if (b == "sigma" and state == "S2") else ""
            print(f"{b:8s} {state:6s} | {absrho.mean():11.3f} {absrho.max():10.3f}  {verdict}{marker}")
            rows.append(dict(comparison="psd_osc_vs_psd", feature=f"psd_osc_{b}",
                             state=state, abs_rho_mean=float(absrho.mean()),
                             abs_rho_max=float(absrho.max()), redundant=redundant))

    if rows:
        out = args.save_path / "results" / "feature_redundancy.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nCSV : {out}")

    print("\n=== Interprétation ===")
    print("  |rho| > 0.8 (REDONDANT) : la feature recopie surtout la pente 1/f ou la PSD")
    print("    brute -> ne pas la garder comme feature séparée dans le papier.")
    print("  |rho| < 0.8 (info ajoutée) : capture autre chose -> à conserver et tester")
    print("    en accuracy marginale via classify.py.")
    print("  Cas fuseaux (psd_osc_sigma en S2) DÉCORRÉLÉ = argument fort pour FOOOF.")


if __name__ == "__main__":
    main()