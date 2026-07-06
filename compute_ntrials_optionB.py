"""Calcule le n_trials_min reel qu'on obtiendrait avec l'option B :
potato par-feature (deja calcule dans dream_features_noica_1000hz_overlap_potato/),
mais n_trials_min pris comme le MINIMUM GLOBAL a travers TOUTES les 6 features
matricielles (pas seulement cov comme le fait classify.py actuellement),
pour les 4 etats de classification (S2, SWS, REM, NREM) apres regroupement
des stades atomiques via CLASSIFICATION_GROUPS -- exactement comme load_subject()
dans classify.py.

Ne lance aucune classification -- juste un comptage, quelques secondes.

Usage:
    python compute_ntrials_optionB.py \
        --potato-dir /home/alouis/scratch/dream_features_noica_1000hz_overlap_potato
"""
import argparse
from pathlib import Path

import numpy as np

KEYS = ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma", "cosp_beta"]
SUBJECTS = [f"{i:02d}" for i in range(1, 39)]

# Reproduit CLASSIFICATION_GROUPS de config_v3.py (confirme dans classify.py / load_subject)
CLASSIFICATION_GROUPS = {
    "S2":   ["S2"],
    "SWS":  ["S3", "S4"],
    "REM":  ["REM"],
    "NREM": ["S1", "S2", "S3", "S4"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--potato-dir", type=Path, required=True)
    return p.parse_args()


def load_atomic(potato_dir: Path, key: str, sub_id: str, atomic_stage: str):
    f = potato_dir / key / f"{key}_s{sub_id}_{atomic_stage}.npz"
    if not f.exists():
        return None
    return np.load(f)["data"]


def load_subject_grouped(potato_dir: Path, key: str, sub_id: str, class_state: str):
    atomic_stages = CLASSIFICATION_GROUPS[class_state]
    parts = [a for s in atomic_stages if (a := load_atomic(potato_dir, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None


def main():
    args = parse_args()

    # ref_counts[(sub_id, class_state)][key] = n_epochs apres potato par-feature
    all_counts = {}
    for class_state in CLASSIFICATION_GROUPS:
        for sub_id in SUBJECTS:
            per_key_counts = {}
            for key in KEYS:
                arr = load_subject_grouped(args.potato_dir, key, sub_id, class_state)
                if arr is not None:
                    per_key_counts[key] = len(arr)
            if per_key_counts:
                all_counts[(sub_id, class_state)] = per_key_counts

    # --- min global a travers TOUTES les features (option B) ---
    all_values_optionB = [
        n for counts in all_counts.values() for n in counts.values()
    ]
    n_trials_optionB = min(all_values_optionB)

    # --- pour comparaison : min global sur cov seulement (methode actuelle, buggee) ---
    cov_values = [
        counts["cov"] for counts in all_counts.values() if "cov" in counts
    ]
    n_trials_cov_only = min(cov_values)

    # --- ou est le pire cas (option B) ---
    worst = min(
        ((sub_id, class_state, key, n)
         for (sub_id, class_state), counts in all_counts.items()
         for key, n in counts.items()),
        key=lambda x: x[3]
    )

    print(f"n_trials_min (methode actuelle, cov seul)      = {n_trials_cov_only}")
    print(f"n_trials_min (option B, min sur les 6 features) = {n_trials_optionB}")
    print(f"Pire cas option B : sub-{worst[0]} / {worst[1]} / {worst[2]} = {worst[3]} epochs")
    print()
    print(f"Perte par rapport a 61 (baseline sans potato) : "
          f"{61 - n_trials_optionB} epochs ({100*(61-n_trials_optionB)/61:.1f}%)")


if __name__ == "__main__":
    main()
