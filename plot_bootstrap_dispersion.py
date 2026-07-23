"""Accuracy dispersion across the 1,000 bootstraps: violin plots by feature x stage.

The barplot reduces each feature to a mean and a standard deviation, which assumes
a symmetrical distribution. These violin plots show the actual shape: asymmetry,
bimodality, tails. A feature whose bootstrap distribution is bimodal does not offer
the same reliability as a tightly clustered feature with an equal mean.

Violin plots rather than stacked histograms: with 24 combinations, histograms would
be unreadable, and comparing features is the core purpose of this figure. For a
diagnostic on a SINGLE feature, a histogram remains clearer, which is handled by
plot_perm_null.py, which overlays the null distribution.

Be careful not to confuse this with plot_perm_null.py: here, the distribution is that of
the ACTUAL accuracies (variability due to epoch sub-sampling), whereas there, it is
that of accuracies under PERMUTED labels (null distribution). The former indicates
"how stable my estimation is", while the latter indicates "how much it exceeds chance".

Usage:
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
    print(f"=== bootstrap dispersion, family {args.family} ===")

    fig, axes = plt.subplots(1, len(STATES_ORDERED),
                             figsize=(3.4 * len(STATES_ORDERED), 4.5),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, state in zip(axes, STATES_ORDERED):
        dists, labels, colors = [], [], []
        for key in keys:
            d = load_result(args.save_path, key, state)
            if d is None:
                print(f"  missing: {key}_{state}.npz")
                continue
            scores = np.asarray(d["acc_scores"])
            if not is_matrix_key(key):
                # Vector feature: we track the best electrode, as done everywhere
                # else, otherwise we would mix 19 distributions.
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

        # Median + IQR overlay: the violin alone does not provide a numeric reference.
        for i, dist in enumerate(dists, start=1):
            q1, med, q3 = np.percentile(dist, [25, 50, 75])
            ax.plot([i, i], [q1, q3], color="k", lw=3, solid_capstyle="butt")
            ax.plot(i, med, "o", color="w", ms=4, zorder=4)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
        ax.set_title(state, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy per bootstrap (%)")

    fig.suptitle(
        f"Dispersion across the 1000 bootstraps, family {args.family}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"bootstrap_dispersion_{args.family}.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()