"""Visualisation UMAP colorée HR vs LR.

Même pipeline que visualize_umap.py mais la couleur encode le groupe
(HR = High Recaller, LR = Low Recaller) au lieu du stade de sommeil.
Génère un panneau : lignes = features, colonnes = stades (S2, SWS, NREM, REM).

Usage :
    python visualize_umap_hrlr.py \\
        --save-path /home/alouis/scratch/dream_features \\
        --overwrite
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from config_v3 import (
    FREQ_DICT,
    FEATURE_KEYS,
    HR_SUBJECTS,
    LR_SUBJECTS,
    EXCLUDED_SUBJECTS,
    CLASSIFICATION_GROUPS,
    STATE_LIST,
)
from utils import load_atomic, upper_tri

# ─── constantes ───────────────────────────────────────────────────────────────

HR_COLOR = "#E84855"   # rouge
LR_COLOR = "#3A86FF"   # bleu
UNKNOWN_COLOR = "#AAAAAA"  # gris (sujets 21/22 exclus)

GROUPS_TO_PLOT = ["S2", "SWS", "NREM", "REM"]

FEATURE_GROUPS = {
    "psd":       [f"psd_{b}"     for b in FREQ_DICT],
    "psd_osc":   [f"psd_osc_{b}" for b in FREQ_DICT],
    "aperiodic": ["aperiodic"],
    "cov":       ["cov"],
    "cosp":      [f"cosp_{b}"    for b in FREQ_DICT],
}

FEATURE_LABELS = {
    "psd":       "PSD brute",
    "psd_osc":   "PSD oscillatoire",
    "aperiodic": "Exposant apériodique",
    "cov":       "Covariance",
    "cosp":      "Cospectrum",
}

ALL_SUBJECTS = [f"{i:02d}" for i in range(1, 39)]


def subject_color(sub_id: str) -> str:
    n = int(sub_id)
    if n in HR_SUBJECTS:   return HR_COLOR
    if n in LR_SUBJECTS:   return LR_COLOR
    return UNKNOWN_COLOR   # 21, 22


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--states", nargs="+", default=GROUPS_TO_PLOT,
                   help="Stades à visualiser (défaut: S2 SWS NREM REM)")
    p.add_argument("--features", nargs="+", default=list(FEATURE_GROUPS),
                   help="Groupes de features à visualiser")
    return p.parse_args()


# ─── chargement ───────────────────────────────────────────────────────────────

def load_state(save_path, key, sub_id, state):
    stages = CLASSIFICATION_GROUPS[state]
    parts = [a for s in stages
             if (a := load_atomic(save_path, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None


def build_vectors(save_path, feat_group_keys, state):
    """Retourne (X, colors, sub_ids_per_epoch) pour un stade et un groupe de features."""
    Xs, colors = [], []

    for sub_id in ALL_SUBJECTS:
        n = int(sub_id)
        if n in EXCLUDED_SUBJECTS:
            continue

        parts = []
        n_epochs = None
        ok = True

        for key in feat_group_keys:
            arr = load_state(save_path, key, sub_id, state)
            if arr is None:
                ok = False
                break
            if key == "cov" or key.startswith("cosp"):
                arr = upper_tri(arr)
            parts.append(arr)
            n_epochs = arr.shape[0]

        if not ok or n_epochs == 0:
            continue

        X_sub = np.concatenate(parts, axis=1)  # (n_epochs, n_features)
        Xs.append(X_sub)
        colors.extend([subject_color(sub_id)] * n_epochs)

    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.array(colors)


# ─── plot ──────────────────────────────────────────────────────────────────────

def plot_hrlr(save_path, states, feature_keys_list, overwrite):
    n_rows = len(feature_keys_list)
    n_cols = len(states)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    legend_handles = [
        mpatches.Patch(color=HR_COLOR, label="HR (High Recallers)"),
        mpatches.Patch(color=LR_COLOR, label="LR (Low Recallers)"),
    ]

    cache_dir = save_path / "_umap_hrlr_cache"
    cache_dir.mkdir(exist_ok=True)

    for row, (fg_name, fg_keys) in enumerate(feature_keys_list):
        for col, state in enumerate(states):
            ax = axes[row, col]
            print(f"  UMAP HR/LR : {fg_name} × {state}")

            cache_file = cache_dir / f"{fg_name}_{state}.npz"
            if cache_file.exists() and not overwrite:
                d = np.load(cache_file)
                emb, colors = d["emb"], d["colors"]
            else:
                X, colors = build_vectors(save_path, fg_keys, state)
                if X is None:
                    ax.text(0.5, 0.5, "données manquantes",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{FEATURE_LABELS[fg_name]}\n{state}")
                    continue
                X_scaled = StandardScaler().fit_transform(X)
                emb = umap.UMAP(
                    n_neighbors=30, min_dist=0.1, random_state=42
                ).fit_transform(X_scaled)
                np.savez_compressed(cache_file, emb=emb, colors=colors)

            # scatter HR puis LR (LR en dessous pour ne pas masquer HR)
            for color, label in [(LR_COLOR, "LR"), (HR_COLOR, "HR")]:
                mask = colors == color
                if mask.any():
                    ax.scatter(emb[mask, 0], emb[mask, 1],
                               c=color, s=4, alpha=0.5, rasterized=True,
                               label=label)

            ax.set_title(f"{FEATURE_LABELS[fg_name]}\n{state}", fontsize=11)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            if row == 0 and col == n_cols - 1:
                ax.legend(handles=legend_handles, markerscale=3, fontsize=9)

    fig.suptitle(
        "UMAP — séparabilité HR vs LR par type de feature et stade de sommeil",
        fontsize=14
    )
    plt.tight_layout()
    out = save_path / "umap_hrlr.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.close()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    states = args.states
    fg_list = [(fg, FEATURE_GROUPS[fg]) for fg in args.features
               if fg in FEATURE_GROUPS]

    print(f"Features : {[f for f, _ in fg_list]}")
    print(f"Stades   : {states}")
    print(f"Sujets   : {len([s for s in ALL_SUBJECTS if int(s) not in EXCLUDED_SUBJECTS])} (hors 21/22)")

    plot_hrlr(args.save_path, states, fg_list, args.overwrite)