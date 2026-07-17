"""Profil accuracy vs bande de fréquence, une courbe par stade.

Répond à "quelle bande porte le signal, et est-ce le même selon le stade ?" —
question que le barplot répond mal (24 barres à comparer de l'oeil) et que la
topomap ne pose pas (elle répond "où", pas "quelle fréquence").

Deux panneaux séparés, jamais superposés : les features matricielles (cospectres)
donnent UN point par bande (accuracy du décodage multivarié 19x19), les
vectorielles (psd) en donnent 19 (une par électrode). Les mettre sur le même axe
ferait croire à des quantités comparables. Le panneau vectoriel trace la
meilleure électrode, avec l'enveloppe min-max des 19 en fond pour montrer la
dispersion spatiale.

La covariance n'apparaît pas : elle est large-bande, donc sans abscisse sur un
axe fréquentiel. Elle est dans le barplot.

Le seuil max-stat pooled est tracé par stade quand --corrected-path est fourni.
Il est constant sur l'axe des bandes (le pooling porte sur toute la famille), ce
qui rend visible d'un coup quelles bandes le franchissent.

Usage :
    python plot_freq_profile.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /scratch/alouis/dream-recall-alex/plot_overlap \
        --alpha 0.05
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    BANDS,
    RESOLUTION,
    STATES_ORDERED,
    load_null_max,
    load_result,
    maxstat_threshold,
)

# Une couleur par stade (Okabe-Ito, colorblind-safe) : ici les courbes sont
# indexées par stade, pas par bande, donc la palette de plot_common ne sert pas.
STATE_COLORS = {"S2": "#0072B2", "SWS": "#009E73", "NREM": "#E69F00", "REM": "#CC79A7"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--corrected-path", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--vector-family", default="psd_classic",
                   choices=["psd_classic", "psd_osc"])
    return p.parse_args()


def matrix_profile(save_path: Path, state: str):
    """Accuracy des cospectres par bande, en % (NaN si absent)."""
    out = []
    for band in BANDS:
        d = load_result(save_path, f"cosp_{band}", state)
        out.append(float(d["acc_mean"]) * 100 if d is not None else np.nan)
    return np.array(out)


def vector_profile(save_path: Path, state: str, prefix: str):
    """(best, min, max) des 19 électrodes par bande, en %.

    best = meilleure électrode : c'est elle qui est testée, et l'enveloppe
    min-max montre à quel point le signal est localisé ou diffus.
    """
    best, lo, hi = [], [], []
    for band in BANDS:
        d = load_result(save_path, f"{prefix}{band}", state)
        if d is None:
            best.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        acc = np.asarray(d["acc_mean"]) * 100
        best.append(acc.max()); lo.append(acc.min()); hi.append(acc.max())
    return np.array(best), np.array(lo), np.array(hi)


def main() -> None:
    args = parse_args()
    prefix = "psd_" if args.vector_family == "psd_classic" else "psd_osc_"
    print(f"=== profil accuracy vs fréquence (subject, p < {args.alpha}) ===")

    x = np.arange(len(BANDS))
    fig, (ax_m, ax_v) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for state in STATES_ORDERED:
        color = STATE_COLORS[state]

        # ── panneau matriciel : cospectres ──
        prof = matrix_profile(args.save_path, state)
        ax_m.plot(x, prof, "-o", color=color, label=state, lw=2, ms=5)

        thr_m = None
        if args.corrected_path:
            nm = load_null_max(args.corrected_path, "matrix", state)
            if nm is not None:
                thr_m = maxstat_threshold(nm, args.alpha) * 100
                ax_m.axhline(thr_m, color=color, ls=":", lw=1, alpha=0.7)
        if thr_m is not None:
            # marque les bandes qui franchissent le seuil de LEUR stade
            sig = prof > thr_m
            ax_m.plot(x[sig], prof[sig], "*", color=color, ms=14,
                      markeredgecolor="k", markeredgewidth=0.5, zorder=5)

        # ── panneau vectoriel : meilleure électrode + enveloppe ──
        best, lo, hi = vector_profile(args.save_path, state, prefix)
        ax_v.plot(x, best, "-o", color=color, label=state, lw=2, ms=5)
        ax_v.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)

        thr_v = None
        if args.corrected_path:
            nm = load_null_max(args.corrected_path, args.vector_family, state)
            if nm is not None:
                thr_v = maxstat_threshold(nm, args.alpha) * 100
                ax_v.axhline(thr_v, color=color, ls=":", lw=1, alpha=0.7)
        if thr_v is not None:
            sig = best > thr_v
            ax_v.plot(x[sig], best[sig], "*", color=color, ms=14,
                      markeredgecolor="k", markeredgewidth=0.5, zorder=5)

    for ax, title in ((ax_m, "Cospectres (riemannien, 19x19)"),
                      (ax_v, f"{args.vector_family} (meilleure électrode)")):
        ax.set_xticks(x)
        ax.set_xticklabels(BANDS)
        ax.set_xlabel("Bande de fréquence")
        ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    ax_m.set_ylabel("Decoding accuracy (%)")
    ax_v.text(0.5, 0.02, "zone claire : min–max sur les 19 électrodes",
              transform=ax_v.transAxes, ha="center", fontsize=8, color="0.4")

    fig.suptitle(
        f"Profil fréquentiel du décodage HR vs LR — permutation sujet (RFX)\n"
        f"···· seuil max-stat pooled par stade   |   * : p < {args.alpha} corrigé",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"freq_profile_{args.vector_family}_subject_p{args.alpha}.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
