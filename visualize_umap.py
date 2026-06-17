"""Visualisation UMAP des features de sommeil.

Lit les tableaux atomiques cachés produits par feat_extract_umap_fooof_v3.py
et génère un panneau 2x3 UMAP coloré par stade de sommeil.

Peut être lancé indépendamment de feat_extract (crash-safe : si les .npz
atomiques existent sur disque, pas besoin de relancer l'extraction complète).

Usage :
    python visualize_umap.py \\
        --save-path /path/to/dream_features \\
        --overwrite  # optionnel : recalcule les vecteurs UMAP même si déjà cachés
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from config import (
    FREQ_DICT,
    FEATURE_KEYS,
    SUBJECT_IDS,
    UMAP_GROUPS,
    UMAP_STATES,
    UMAP_COLORS,
)
from utils import load_atomic, upper_tri


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Dossier contenant les .npz atomiques (sortie de feat_extract)")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Recalcule les vecteurs UMAP même si umap_vectors.npz existe")
    return p.parse_args()


# ─── groupements de features pour les 6 panneaux UMAP ────────────────────────

UMAP_FEATURE_GROUPS: dict[str, list[str]] = {
    "psd":       [f"psd_{b}"     for b in FREQ_DICT],
    "psd_osc":   [f"psd_osc_{b}" for b in FREQ_DICT],
    "cov":       ["cov"],
    "cosp":      [f"cosp_{b}"    for b in FREQ_DICT],
    "aperiodic": ["aperiodic"],
}

UMAP_PANEL_TITLES: dict[str, str] = {
    "psd":       "PSD (bandes brutes)",
    "psd_osc":   "PSD oscillatoire (corrigée 1/f)",
    "cov":       "Covariance",
    "cosp":      "Cospectrum (toutes bandes)",
    "aperiodic": "Exposant apériodique",
    "all":       "Toutes features combinées",
}


# ─── construction des vecteurs ────────────────────────────────────────────────

def build_umap_vectors(
    save_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Construit les vecteurs UMAP depuis les .npz atomiques cachés.

    Pour chaque sujet x état UMAP, charge les tableaux atomiques
    correspondants, aplatit les matrices (cov/cosp -> triangle supérieur),
    et concatène par groupe de features.

    Returns
    -------
    vectors : dict[group_name] -> (n_total_epochs, n_features)
    labels  : (n_total_epochs,) stade UMAP de chaque epoch
    """
    vectors: dict[str, list[np.ndarray]] = {g: [] for g in UMAP_FEATURE_GROUPS}
    labels: list[str] = []

    for sub_id in SUBJECT_IDS:
        for state, stages in UMAP_GROUPS.items():
            per_key: dict[str, np.ndarray] = {}
            n_epochs = None
            ok = True

            for key in FEATURE_KEYS:
                parts = [
                    a for s in stages
                    if (a := load_atomic(save_path, key, sub_id, s)) is not None
                ]
                if not parts:
                    ok = False
                    break
                arr = np.concatenate(parts, axis=0)
                if key == "cov" or key.startswith("cosp"):
                    arr = upper_tri(arr)
                per_key[key] = arr
                n_epochs = arr.shape[0]

            if not ok:
                continue

            for group, keys in UMAP_FEATURE_GROUPS.items():
                vectors[group].append(
                    np.concatenate([per_key[k] for k in keys], axis=1)
                )
            labels.extend([state] * n_epochs)

    out = {g: np.concatenate(v, axis=0) for g, v in vectors.items()}
    out["all"] = np.concatenate([out[g] for g in UMAP_FEATURE_GROUPS], axis=1)
    return out, np.array(labels)


def build_umap_vectors_cached(
    save_path: Path, overwrite: bool = False
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Idem build_umap_vectors mais avec cache disque (umap_vectors.npz).

    Évite de relire tous les .npz atomiques si seul le plot a crashé.
    """
    cache = save_path / "umap_vectors.npz"
    if cache.exists() and not overwrite:
        d = np.load(cache, allow_pickle=True)
        vectors = {k: d[k] for k in d.files if k != "labels"}
        return vectors, d["labels"]

    vectors, labels = build_umap_vectors(save_path)
    np.savez_compressed(cache, labels=labels, **vectors)
    return vectors, labels


# ─── visualisation ────────────────────────────────────────────────────────────

def plot_umaps(
    vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: Path,
) -> None:
    """Panneau 2x3 : un subplot par groupe de features, coloré par stade UMAP."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for ax, (fname, title) in zip(axes.flatten(), UMAP_PANEL_TITLES.items()):
        print(f"  UMAP: {fname}")
        X = StandardScaler().fit_transform(vectors[fname])
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(X)

        for state in UMAP_STATES:
            mask = labels == state
            if mask.any():
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=UMAP_COLORS[state], s=3, alpha=0.4, rasterized=True,
                )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(
            handles=[mpatches.Patch(color=UMAP_COLORS[s], label=s) for s in UMAP_STATES],
            markerscale=3,
        )

    fig.suptitle(
        "UMAP — séparabilité des stades de sommeil par type de feature", fontsize=15
    )
    plt.tight_layout()
    out = save_path / "umap_sleep_stages.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("=== construction des vecteurs UMAP ===")
    vectors, labels = build_umap_vectors_cached(args.save_path, args.overwrite)

    print("=== projection et plot ===")
    plot_umaps(vectors, labels, args.save_path)
