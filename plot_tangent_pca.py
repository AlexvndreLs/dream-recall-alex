#!/usr/bin/env python3
"""
Projection tangente + PCA 2D des features matricielles significatives (HR vs LR).

Un point = un sujet. Pour chaque sujet et chaque etat, les epochs sont resumes
par leur moyenne riemannienne, ce qui donne 36 SPD matrices 19x19. Ces matrices
sont projetees dans l'espace tangent a la moyenne riemannienne globale, puis
reduites a 2D par PCA. Ellipses de covariance a 2 sigma par groupe.

ATTENTION A L'INTERPRETATION
----------------------------
La PCA est NON SUPERVISEE : elle cherche les axes de variance maximale, pas les
axes discriminants. En EEG ces axes coincident rarement. Une absence de
separation visuelle sur cette figure NE CONTREDIT PAS les accuracies
significatives obtenues par LDA / MDM, qui exploitent des directions que la PCA
peut placer sur des composantes d'ordre eleve. Cette figure est descriptive,
elle n'a aucune valeur inferentielle et ne doit pas etre citee comme resultat.

Le niveau sujet (et non epoch) est choisi deliberement : au niveau epoch, la
variance inter-sujets domine largement la variance HR/LR, et les clusters
observes refleteraient l'identite des sujets, pas leur groupe.

Usage
-----
    python3 plot_tangent_pca.py
    python3 plot_tangent_pca.py --features cosp_sigma/S2 cosp_delta/SWS
    python3 plot_tangent_pca.py --data-root /scratch/alouis/... --out-dir ...
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from config_v3 import CLASSIFICATION_GROUPS, HR_SUBJECTS, LR_SUBJECTS

try:
    # pyriemann.utils.mean est deprecie et disparait en 0.14 au profit de
    # pyriemann.geometry.mean. On tente le nouvel emplacement d'abord pour
    # rester compatible avec les deux, sans emettre de DeprecationWarning.
    try:
        from pyriemann.geometry.mean import mean_riemann
    except ImportError:
        from pyriemann.utils.mean import mean_riemann
    from pyriemann.tangentspace import TangentSpace
except ImportError as exc:
    sys.exit(f"pyriemann requis : {exc}")


DEFAULT_DATA_ROOT = Path("/scratch/alouis/dream_features_noica_1000hz_overlap")
DEFAULT_OUT_DIR = Path("plot_overlap")

# Les 4 combinaisons significatives apres correction maxstat poolee (p < 0.05)
DEFAULT_FEATURES = [
    "cosp_sigma/S2",
    "cosp_delta/SWS",
    "cosp_delta/NREM",
    "cosp_alpha/S2",
]

# Okabe-Ito, coherent avec UMAP_COLORS de config_v3
COLOR_HR = "#0072B2"
COLOR_LR = "#D55E00"


def analysis_subjects():
    """Numeros des 36 sujets inclus, tries, avec leurs labels (1 = HR, 0 = LR)."""
    subs = sorted(HR_SUBJECTS | LR_SUBJECTS)
    labels = np.array([1 if s in HR_SUBJECTS else 0 for s in subs], dtype=int)
    return subs, labels


def load_subject_state(data_root, feature, subject, state):
    """Charge et concatene les epochs d'un sujet pour un etat de classification.

    Les etats composites (SWS, NREM) sont obtenus par concatenation des stades
    atomiques, conformement a CLASSIFICATION_GROUPS. Aucun recalcul.

    Retourne un array (n_epochs, 19, 19), ou None si aucune donnee.
    """
    if state not in CLASSIFICATION_GROUPS:
        raise KeyError(
            f"etat '{state}' inconnu, attendus : {list(CLASSIFICATION_GROUPS)}"
        )

    blocks = []
    for atomic in CLASSIFICATION_GROUPS[state]:
        path = data_root / feature / f"{feature}_s{subject:02d}_{atomic}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as npz:
            if "data" not in npz.files:
                raise KeyError(f"cle 'data' absente de {path}, cles : {npz.files}")
            arr = np.asarray(npz["data"], dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
            raise ValueError(f"shape inattendue {arr.shape} dans {path}")
        if arr.shape[0] > 0:
            blocks.append(arr)

    if not blocks:
        return None
    return np.concatenate(blocks, axis=0)


def regularize(mats, eps=1e-10):
    """Ridge minimal sur la diagonale pour garantir la definie-positivite.

    Les cospectres estimes sur peu d'epochs peuvent etre mal conditionnes, ce
    qui fait echouer les logarithmes matriciels. Le shift est proportionnel a
    la trace pour rester invariant a l'echelle.
    """
    n = mats.shape[-1]
    eye = np.eye(n)
    trace = np.trace(mats, axis1=-2, axis2=-1)[:, None, None] / n
    return mats + eps * trace * eye


def subject_means(data_root, feature, state, subjects):
    """Moyenne riemannienne des epochs de chaque sujet.

    Retourne (means, kept_idx, n_epochs) ou means est (n_kept, 19, 19).
    Les sujets sans donnee pour cet etat sont exclus explicitement, avec un
    avertissement, plutot que remplaces par une valeur par defaut.
    """
    means, kept, counts = [], [], []
    for i, sub in enumerate(subjects):
        epochs = load_subject_state(data_root, feature, sub, state)
        if epochs is None:
            print(f"  [skip] s{sub:02d} : aucune epoch pour {feature}/{state}")
            continue
        epochs = regularize(epochs)
        means.append(mean_riemann(epochs))
        kept.append(i)
        counts.append(epochs.shape[0])
    if not means:
        return None, None, None
    return np.stack(means), np.array(kept, dtype=int), np.array(counts, dtype=int)


def tangent_pca(means):
    """Espace tangent a la moyenne globale, puis PCA 2D.

    Retourne (coords (n, 2), ratio de variance expliquee (2,)).
    """
    ts = TangentSpace(metric="riemann")
    vecs = ts.fit_transform(means)

    centered = vecs - vecs.mean(axis=0, keepdims=True)
    # SVD plutot que sklearn.PCA : n_samples (36) << n_features (190), la SVD
    # economique est exacte et evite une dependance supplementaire.
    _, sing, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    var = sing**2
    ratio = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    return coords, ratio


def covariance_ellipse(ax, points, color, n_std=2.0):
    """Ellipse de covariance a n_std ecarts-types, tracee sur ax."""
    if points.shape[0] < 3:
        return
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if np.any(vals <= 0):
        return

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ax.add_patch(
        Ellipse(
            xy=points.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=1.5,
            linestyle="--",
            zorder=1,
        )
    )


def plot_panel(ax, coords, labels, ratio, title, counts=None):
    """Trace un panneau : scatter HR/LR, ellipses, centroides."""
    for lab, color, name in ((1, COLOR_HR, "HR"), (0, COLOR_LR, "LR")):
        pts = coords[labels == lab]
        if pts.shape[0] == 0:
            continue
        covariance_ellipse(ax, pts, color)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            s=45,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=f"{name} (n={pts.shape[0]})",
            zorder=3,
        )
        ax.scatter(
            *pts.mean(axis=0),
            c=color,
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )

    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f} % var.)")
    ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f} % var.)")

    subtitle = title
    if counts is not None and counts.size > 0:
        subtitle += f"\nepochs/sujet : med {int(np.median(counts))}"
        subtitle += f" [{counts.min()}, {counts.max()}]"
    ax.set_title(subtitle, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="combinaisons 'feature/etat', ex. cosp_sigma/S2",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        sys.exit(f"repertoire introuvable : {args.data_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subjects, labels_all = analysis_subjects()
    print(f"{len(subjects)} sujets : {int(labels_all.sum())} HR, "
          f"{int((labels_all == 0).sum())} LR")

    results = []
    for combo in args.features:
        if "/" not in combo:
            sys.exit(f"format attendu 'feature/etat', recu '{combo}'")
        feature, state = combo.split("/", 1)
        print(f"\n{combo}")

        if not (args.data_root / feature).is_dir():
            sys.exit(f"feature introuvable : {args.data_root / feature}")

        means, kept, counts = subject_means(args.data_root, feature, state, subjects)
        if means is None:
            print(f"  [skip] aucune donnee pour {combo}")
            continue

        coords, ratio = tangent_pca(means)
        labels = labels_all[kept]
        print(f"  {means.shape[0]} sujets retenus, "
              f"PC1 {100 * ratio[0]:.1f} %, PC2 {100 * ratio[1]:.1f} %")

        fig, ax = plt.subplots(figsize=(5.5, 5))
        plot_panel(ax, coords, labels, ratio, combo, counts)
        fig.tight_layout()
        stem = combo.replace("/", "_")
        path = args.out_dir / f"tangent_pca_{stem}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")

        results.append((combo, coords, labels, ratio, counts))

    if len(results) > 1:
        ncols = 2
        nrows = int(np.ceil(len(results) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (combo, coords, labels, ratio, counts) in zip(axes, results):
            plot_panel(ax, coords, labels, ratio, combo, counts)
        for ax in axes[len(results):]:
            ax.set_visible(False)
        fig.suptitle(
            "Espace tangent riemannien, PCA 2D (niveau sujet)\n"
            "PCA non supervisee : figure descriptive, sans valeur inferentielle",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = args.out_dir / "tangent_pca_grid.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()