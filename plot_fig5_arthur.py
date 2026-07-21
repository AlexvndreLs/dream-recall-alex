"""Plot Fig. 5 (these Arthur chap.1) : pie charts (selection rate) + barplot (accuracy).

Consomme fig5_roi_{state}.npz (aggregate_roi_fig5.py). Reproduit la Fig.5 :
  - GAUCHE : un pie chart par ROI, repartition des bandes selectionnees par l'EFS.
             Seules les COMBINAISONS depassant le seuil SR (25%) sont colorees
             (legende Fig.5) ; le reste en gris.
  - DROITE : barplot de l'accuracy holdout par ROI, ligne pointillee au seuil p<0.001.

Usage
-----
    python plot_fig5_arthur.py \\
        --in-dir /scratch/alouis/dream_features_noica_1000hz_corrected/fig5_recompute \\
        --state  S2 \\
        --out    fig5_S2.png

    # variante montage 11 electrodes d'Arthur :
    #   --npz-name fig5_roi_arthur11_S2.npz
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config_v3 import FREQ_DICT

BANDS = list(FREQ_DICT)
# couleurs par bande (coherentes pie/barplot)
BAND_COLORS = {
    "delta": "#4C72B0",
    "theta": "#55A868",
    "alpha": "#C44E52",
    "sigma": "#8172B3",
    "beta":  "#CCB974",
}
GREY = "#CCCCCC"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--state",  type=str, default="S2")
    p.add_argument("--npz-name", type=str, default=None,
                   help="Nom du .npz (defaut fig5_roi_{state}.npz). Pour la variante "
                        "11-elec : fig5_roi_arthur11_{state}.npz.")
    p.add_argument("--chance", type=float, default=0.5,
                   help="Niveau de chance (ligne de reference barplot).")
    p.add_argument("--out", type=Path, default=Path("fig5.png"))
    return p.parse_args()


def load_roi_data(path):
    d = np.load(path, allow_pickle=True)
    roi_data = d["roi_data"][0]        # dict ROI -> stats
    rois = [r.decode() if isinstance(r, bytes) else str(r) for r in d["rois"]]
    sr_threshold = float(d["sr_threshold"])
    return roi_data, rois, sr_threshold


def main():
    args = parse_args()
    name = args.npz_name or f"fig5_roi_{args.state}.npz"
    path = args.in_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    roi_data, rois, sr_thr = load_roi_data(path)

    n_roi = len(rois)
    fig, axes = plt.subplots(n_roi, 2, figsize=(9, 2.1 * n_roi),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    axes = np.atleast_2d(axes)

    for r, roi in enumerate(rois):
        stats = roi_data[roi]
        band_counts = np.asarray(stats["band_counts"], dtype=float)
        band_rates = np.asarray(stats["band_rates"], dtype=float)
        acc = float(stats["holdout_acc"])
        frac_sig = stats.get("frac_sig", np.nan)

        # --- PIE (gauche) : repartition des bandes ; grise celles < seuil SR
        ax_pie = axes[r, 0]
        colors, labels, sizes = [], [], []
        for i, b in enumerate(BANDS):
            if band_counts[i] <= 0:
                continue
            sizes.append(band_counts[i])
            # colore si le taux de selection de la bande depasse le seuil, sinon gris
            above = band_rates[i] >= sr_thr
            colors.append(BAND_COLORS[b] if above else GREY)
            labels.append(f"{b} {band_rates[i]*100:.0f}%" if above else "")
        ax_pie.pie(sizes, colors=colors, labels=labels, startangle=90,
                   textprops={"fontsize": 7},
                   wedgeprops={"edgecolor": "white", "linewidth": 0.5})
        ax_pie.set_title(roi, fontsize=10, weight="bold")

        # --- BARPLOT (droite) : accuracy holdout
        ax_bar = axes[r, 1]
        # couleur de la barre = bande dominante
        top_band = BANDS[int(np.argmax(band_counts))]
        ax_bar.barh([0], [acc * 100], color=BAND_COLORS[top_band], height=0.5)
        ax_bar.axvline(args.chance * 100, color="grey", ls="-", lw=0.8)  # chance 50%
        ax_bar.set_xlim(40, 75)
        ax_bar.set_yticks([])
        fs = "" if (isinstance(frac_sig, float) and np.isnan(frac_sig)) \
            else f"  frac_sig={frac_sig:.2f}"
        ax_bar.set_title(f"acc = {acc*100:.1f}%  (top: {top_band}){fs}", fontsize=9)
        ax_bar.set_xlabel("Holdout accuracy (%)" if r == n_roi - 1 else "")

    # ligne significativite p<0.001 : note globale (le seuil exact depend du plot,
    # ici on affiche la chance ; le seuil p<0.001 par ROI est dans frac_sig/p_median)
    fig.suptitle(f"Fig. 5 (Arthur chap.1) — EFS holdout par ROI, {args.state}\n"
                 f"(pie: selection rate, seuil SR {int(sr_thr*100)}% ; "
                 f"barplot: accuracy holdout, ligne = chance 50%)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {args.out}")


if __name__ == "__main__":
    main()