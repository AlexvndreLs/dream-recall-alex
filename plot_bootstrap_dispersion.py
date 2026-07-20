"""Dispersion des accuracies sur les 1000 bootstraps : violons par feature x stade.

Le barplot résume chaque feature à une moyenne et un écart-type, ce qui suppose
une distribution symétrique. Ces violons montrent la forme réelle : asymétrie,
bimodalité, queues. Une feature dont la distribution bootstrap est bimodale n'a
pas la même fiabilité qu'une feature bien concentrée, à moyenne égale.

Violons plutôt qu'histogrammes empilés : à 24 combos, des histogrammes seraient
illisibles, et la comparaison entre features est l'objet même de la figure. Pour
un diagnostic sur UNE feature, l'histogramme reste plus lisible, c'est ce que
fait plot_perm_null.py, qui superpose la nulle.

Attention à ne pas confondre avec plot_perm_null.py : ici la distribution est
celle des accuracies RÉELLES (variabilité due au sous-échantillonnage des
epochs), là-bas celle des accuracies sous labels PERMUTÉS (loi nulle). La
première dit "à quel point mon estimation est stable", la seconde "à quel point
elle dépasse le hasard".

Usage :
    python plot_bootstrap_dispersion.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir   /scratch/alouis/dream-recall-alex/plot_overlap \
        --family matrix
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    FAMILY_KEYS,
    RESOLUTION,
    STATES_ORDERED,
    band_label,
    is_matrix_key,
    key_color,
    load_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--family", default="matrix", choices=list(FAMILY_KEYS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keys = FAMILY_KEYS[args.family]
    print(f"=== dispersion bootstrap, famille {args.family} ===")

    fig, axes = plt.subplots(1, len(STATES_ORDERED),
                             figsize=(3.4 * len(STATES_ORDERED), 4.5),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, state in zip(axes, STATES_ORDERED):
        dists, labels, colors = [], [], []
        for key in keys:
            d = load_result(args.save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                continue
            scores = np.asarray(d["acc_scores"])
            if not is_matrix_key(key):
                # feature vectorielle : on suit la meilleure électrode, comme
                # partout ailleurs, sinon on mélangerait 19 distributions.
                best = int(np.asarray(d["acc_mean"]).argmax())
                scores = scores[:, best]
            dists.append(scores * 100)
            labels.append(band_label(key))
            colors.append(key_color(key))

        if not dists:
            ax.axis("off")
            continue

        parts = ax.violinplot(dists, showextrema=False, widths=0.85)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
            body.set_edgecolor("0.3")
            body.set_linewidth(0.5)

        # Médiane + IQR par-dessus : le violon seul ne donne pas de repère chiffré.
        for i, dist in enumerate(dists, start=1):
            q1, med, q3 = np.percentile(dist, [25, 50, 75])
            ax.plot([i, i], [q1, q3], color="k", lw=3, solid_capstyle="butt")
            ax.plot(i, med, "o", color="w", ms=4, zorder=4)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
        ax.set_title(state, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy sur chaque bootstrap (%)")

    fig.suptitle(
        f"Dispersion des 1000 bootstraps, famille {args.family}\n"
        f"distribution des accuracies RÉELLES (variabilité du sous-échantillonnage), "
        f"à ne pas confondre avec la loi nulle par permutation",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"bootstrap_dispersion_{args.family}.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()