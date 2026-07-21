"""Plot Fig. 5 VERSION CODE PUBLIC d'Arthur : pie chart simple a 5 BANDES.

REPRODUIT EXACTEMENT visu_piecharts_fselect.py du repo public arthurdehgan/sleep.
Contrairement a plot_fig5_arthur_grid.py (qui reconstruit la figure de la THESE avec
combinaisons + parts grises, code NON publie), ce script suit le code PUBLIC, qui fait
un pie a 5 parts (une par bande), SANS combinaisons, SANS gris, SANS seuil.

Fidelite ligne par ligne a visu_piecharts_fselect.py :
- FREQS = ["Delta","Theta","Alpha","Sigma","Beta"] (ordre exact).
- super_count sur les bandes APLATIES (chaque bande d'un sous-ensemble compte +1).
  -> c'est notre band_counts (deja stocke par aggregate_roi_fig5.py).
- pie a 5 parts : plt.pie([count[freq] for freq in FREQS]).
- grille : 1ere ligne "All" (pie global toutes electrodes) + 1 ligne par ROI ;
  colonnes = stades. (ROI en lignes, stades en colonnes.)
- pas de couleurs specifiees chez Arthur (cycle matplotlib) ; ici on garde des couleurs
  par bande pour la lisibilite, mais la STRUCTURE (5 parts) est identique.
- pas de barplot, pas de seuil, pas de gris (le code public n'en a pas).

Sources : fig5_roi_{state}.npz par stade (band_counts). Un fichier par stade.

Usage
-----
    python plot_fig5_arthur_public.py \\
        --in-dir /scratch/alouis/dream_features_noica_1000hz_corrected/fig5_recompute \\
        --out    fig5_public_5bandes.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config_v3 import FREQ_DICT

# ordre EXACT d'Arthur (FREQS dans visu_piecharts_fselect.py)
BANDS = list(FREQ_DICT)                    # delta, theta, alpha, sigma, beta
STAGES = ["S2", "SWS", "NREM", "REM"]
ROIS_ORDER = ["prefrontal", "fronto-central", "temporal", "centro-parietal", "occipital"]
BAND_COLORS = {
    "delta": "#4C72B0", "theta": "#DD8452", "alpha": "#55A868",
    "sigma": "#C44E52", "beta": "#8172B3",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("fig5_public_5bandes.png"))
    return p.parse_args()


def load_stage(in_dir, stage):
    path = in_dir / f"fig5_roi_{stage}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return d["roi_data"][0]


def main():
    args = parse_args()

    # grille : lignes = 1 "All" + 5 ROI = 6 ; colonnes = 4 stades (comme Arthur 6x4)
    nrow = 1 + len(ROIS_ORDER)
    fig, axes = plt.subplots(nrow, len(STAGES), figsize=(10, 13))
    axes = np.atleast_2d(axes)
    colors = [BAND_COLORS[b] for b in BANDS]

    for j, stage in enumerate(STAGES):
        roi_data = load_stage(args.in_dir, stage)
        if roi_data is None:
            for i in range(nrow):
                axes[i, j].axis("off")
            axes[0, j].set_title(f"{stage}\n(non calcule)", fontsize=9)
            continue

        # ligne 0 : "All" = somme des band_counts sur TOUS les ROI (toutes electrodes)
        all_counts = np.zeros(len(BANDS))
        for roi in ROIS_ORDER:
            if roi in roi_data:
                all_counts += np.asarray(roi_data[roi]["band_counts"], dtype=float)
        axes[0, j].pie(all_counts, colors=colors, startangle=90,
                       wedgeprops={"edgecolor": "white", "linewidth": 0.4})
        axes[0, j].set_title(stage, fontsize=12, weight="bold")
        if j == 0:
            axes[0, j].set_ylabel("All", fontsize=11, rotation=90, labelpad=20)

        # lignes 1..5 : un pie par ROI (5 bandes)
        for i, roi in enumerate(ROIS_ORDER, start=1):
            ax = axes[i, j]
            if roi not in roi_data:
                ax.axis("off")
                continue
            bc = np.asarray(roi_data[roi]["band_counts"], dtype=float)
            ax.pie(bc, colors=colors, startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 0.4})
            if j == 0:
                ax.set_ylabel(roi, fontsize=10, rotation=90, labelpad=20)

    # legende des 5 bandes (comme Arthur, en bas)
    handles = [Patch(facecolor=BAND_COLORS[b], label=b.capitalize()) for b in BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=len(BANDS),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Fig. 5 (Arthur chap.1) — VERSION CODE PUBLIC : selection rate par "
                 "BANDE (5 parts, pas de combinaisons)", fontsize=11, y=1.0)
    fig.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {args.out}")


if __name__ == "__main__":
    main()