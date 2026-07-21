"""Plot Fig. 5 VERSION ARTHUR (these chap.1) : grille 4 stades x 5 ROI + barplots.

Reproduit la mise en forme EXACTE de la Fig.5 d'Arthur (image 4) :
  - Grille : 4 lignes (stades S2, SWS, NREM, REM) x 5 colonnes (ROI).
  - Chaque cellule = un PIE CHART montrant le selection rate des COMBINAISONS de
    bandes (pas juste les bandes isolees). Seules les combinaisons > seuil SR (25%)
    sont colorees ; le reste en gris ("others").
  - Colonne de droite : un BARPLOT par stade, accuracy holdout (DA %) par ROI, barre
    coloree selon la combinaison dominante, ligne pointillee = seuil de significativite.

Sources : fig5_roi_{state}.npz par stade (aggregate_roi_fig5.py), un par stade dans
--in-dir. Les stades absents (non calcules) affichent une ligne vide.

Legende des combinaisons (couleurs fideles a Arthur, image 4) :
  delta, theta, alpha, sigma, beta, delta+sigma, beta+sigma, sigma+theta,
  beta+sigma+theta, alpha+delta+theta, others (gris).

Usage
-----
    python plot_fig5_arthur_grid.py \\
        --in-dir /scratch/alouis/dream_features_noica_1000hz_corrected/fig5_recompute \\
        --sr-threshold 0.25 --sig-line 52 \\
        --out fig5_arthur_grid.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config_v3 import FREQ_DICT

BANDS = list(FREQ_DICT)                       # delta theta alpha sigma beta
STAGES = ["S2", "SWS", "NREM", "REM"]
ROIS_ORDER = ["prefrontal", "fronto-central", "temporal", "centro-parietal", "occipital"]
ROI_SHORT = {"prefrontal": "PF", "fronto-central": "FC", "temporal": "TP",
             "centro-parietal": "CP", "occipital": "OC"}
SYM = {"delta": "δ", "theta": "θ", "alpha": "α", "sigma": "σ", "beta": "β"}

# couleurs des combinaisons (fideles a la legende d'Arthur, image 4)
COMBO_COLORS = {
    ("delta",): "#4C72B0",
    ("theta",): "#DD8452",
    ("alpha",): "#55A868",
    ("sigma",): "#C44E52",
    ("beta",): "#8172B3",
    ("delta", "sigma"): "#4C72B0",
    ("beta", "sigma"): "#CCB974",
    ("sigma", "theta"): "#2CA02C",
    ("beta", "sigma", "theta"): "#D62728",
    ("alpha", "delta", "theta"): "#E377C2",
}
GREY = "#CCCCCC"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--sr-threshold", type=float, default=0.25)
    p.add_argument("--sig-line", type=float, default=52.0,
                   help="Position (%) de la ligne pointillee de significativite.")
    p.add_argument("--out", type=Path, default=Path("fig5_arthur_grid.png"))
    return p.parse_args()


def combo_key(combo):
    """Normalise une combinaison en tuple trie de noms de bandes."""
    return tuple(sorted(combo))


def load_stage(in_dir, stage):
    """Charge fig5_roi_{stage}.npz -> roi_data dict, ou None si absent."""
    path = in_dir / f"fig5_roi_{stage}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return d["roi_data"][0]


def combo_label(combo):
    return "+".join(SYM[b] for b in combo)


def main():
    args = parse_args()

    fig, axes = plt.subplots(len(STAGES), len(ROIS_ORDER) + 1,
                             figsize=(16, 3.2 * len(STAGES)),
                             gridspec_kw={"width_ratios": [1] * len(ROIS_ORDER) + [1.3]})
    axes = np.atleast_2d(axes)

    used_combos = set()

    for r, stage in enumerate(STAGES):
        roi_data = load_stage(args.in_dir, stage)
        # label de ligne (stade)
        axes[r, 0].annotate(stage, xy=(-0.35, 0.5), xycoords="axes fraction",
                            fontsize=13, weight="bold", ha="center", va="center",
                            rotation=90)
        if roi_data is None:
            for c in range(len(ROIS_ORDER) + 1):
                axes[r, c].axis("off")
            axes[r, 0].annotate(f"{stage}\n(non calcule)", xy=(0.5, 0.5),
                                xycoords="axes fraction", ha="center", fontsize=9)
            continue

        accs, bar_colors = [], []
        for c, roi in enumerate(ROIS_ORDER):
            ax = axes[r, c]
            if roi not in roi_data:
                ax.axis("off")
                accs.append(np.nan); bar_colors.append(GREY)
                continue
            stats = roi_data[roi]
            combo_rates = stats["combo_rates"]   # dict combo(tuple) -> rate
            # Chaque COMBINAISON = une part (comme Arthur, image reelle Fig.5). Les
            # combinaisons > seuil ET connues sont colorees + labellisees ; TOUTES les
            # autres restent des parts SEPAREES grises (d'ou les multiples subdivisions
            # grises visibles chez Arthur). On NE fusionne PAS le gris.
            sizes, colors, labels = [], [], []
            # trie par rate decroissant pour un rendu propre
            for combo, rate in sorted(combo_rates.items(), key=lambda x: -x[1]):
                ck = combo_key(combo)
                sizes.append(rate)
                if rate >= args.sr_threshold and ck in COMBO_COLORS:
                    colors.append(COMBO_COLORS[ck])
                    labels.append(combo_label(ck))
                    used_combos.add(ck)
                else:
                    colors.append(GREY)
                    labels.append("")            # gris, pas de label, mais part separee
            ax.pie(sizes, colors=colors, labels=labels, startangle=90,
                   textprops={"fontsize": 8},
                   wedgeprops={"edgecolor": "#999999", "linewidth": 0.3})
            # accuracy + couleur barre = combinaison dominante coloree (ou grise)
            acc = float(stats["holdout_acc"]) * 100
            accs.append(acc)
            top = None
            for combo, rate in sorted(combo_rates.items(), key=lambda x: -x[1]):
                ck = combo_key(combo)
                if rate >= args.sr_threshold and ck in COMBO_COLORS:
                    top = ck; break
            bar_colors.append(COMBO_COLORS[top] if top else GREY)
            if r == len(STAGES) - 1:
                ax.set_xlabel(roi.replace("-", "-\n").title(), fontsize=10)

        # barplot du stade (derniere colonne)
        ax_bar = axes[r, len(ROIS_ORDER)]
        xs = np.arange(len(ROIS_ORDER))
        vals = [a if not np.isnan(a) else 0 for a in accs]
        ax_bar.bar(xs, vals, color=bar_colors, width=0.7)
        ax_bar.axhline(args.sig_line, ls="--", color="k", lw=0.8)
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels([ROI_SHORT[r_] for r_ in ROIS_ORDER], fontsize=8)
        ax_bar.set_ylim(20, 67)
        ax_bar.set_ylabel("DA (%)", fontsize=9)

    # legende globale des combinaisons (bas), seulement celles utilisees
    legend_order = [("delta",), ("theta",), ("alpha",), ("sigma",), ("beta",),
                    ("delta", "sigma"), ("beta", "sigma"), ("sigma", "theta"),
                    ("beta", "sigma", "theta"), ("alpha", "delta", "theta")]
    handles = [Patch(facecolor=COMBO_COLORS[ck], label=combo_label(ck))
               for ck in legend_order]
    handles.append(Patch(facecolor=GREY, label="others"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fig. 5 (Arthur chap.1) — EFS holdout par ROI et stade  "
                 f"(pie: selection rate combinaisons, seuil {int(args.sr_threshold*100)}%)",
                 fontsize=12, y=1.0)
    fig.subplots_adjust(hspace=0.25, wspace=0.15, bottom=0.08)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {args.out}")


if __name__ == "__main__":
    main()