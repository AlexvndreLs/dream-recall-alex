#!/usr/bin/env python3
"""Trois figures de synthese sur la correction multiple, schema subject (RFX).

Lit pvalue_summary_table.csv (build_pvalue_summary_table.py), ne touche a
aucun .npz. Colonnes utilisees : p_non_corrige_subject, p_maxstat_arthur_subject,
p_maxstat_pooled_subject.

Le niveau "arthur" (max-stat sur les 19 electrodes) n'existe que pour les
features vectorielles : les features matricielles (cov, cosp_*) produisent une
seule accuracy, pas d'accuracy par electrode. Les panneaux sont donc separes,
3 niveaux a gauche, 2 niveaux a droite, jamais de N/A affiche comme un zero.

Figures produites :
  fig4_n_significatifs_par_niveau.png  barplot du nombre de tests significatifs
  fig5_courbe_survie_pvalues.png       p triees par rang, seuil alpha en ligne
  fig6_espace_de_test.png              schema des deux familles de correction

Usage:
    python plot_correction_summary.py \\
        --csv /scratch/alouis/dream_features_noica_1000hz_overlap/results/pvalue_summary_table.csv \\
        --out-dir ~/dream-recall-alex/plot_overlap
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

COL_RAW = "p_non_corrige_subject"
COL_ARTHUR = "p_maxstat_arthur_subject"
COL_POOLED = "p_maxstat_pooled_subject"

LABELS = {
    COL_RAW: "non corrige",
    COL_ARTHUR: "maxstat electrode\n(Arthur, 19 tests)",
    COL_POOLED: "maxstat pooled\n(famille de features)",
}
COLORS = {COL_RAW: "#b0b0b0", COL_ARTHUR: "#4878a8", COL_POOLED: "#c04040"}

FAMILIES = {
    "matrix": ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma",
               "cosp_beta"],
    "psd_classic": ["psd_delta", "psd_theta", "psd_alpha", "psd_sigma",
                    "psd_beta"],
}


def assign_family(key: str) -> str:
    """Famille de correction pooled, meme logique que build_pvalue_summary_table.py."""
    for name, keys in FAMILIES.items():
        if key in keys:
            return name
    return "isolee"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True,
                   help="pvalue_summary_table.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_table(csv: Path) -> pd.DataFrame:
    """Charge le CSV et convertit les colonnes de p en float.

    Les cellules "N/A (matriciel)" deviennent NaN : elles ne sont ni
    significatives ni non significatives, elles n'existent pas.
    """
    df = pd.read_csv(csv)
    for c in (COL_RAW, COL_ARTHUR, COL_POOLED):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fig_n_significatifs(df: pd.DataFrame, out_dir: Path, alpha: float):
    """Figure 4 : combien de tests survivent a chaque niveau de correction."""
    vec = df[~df["is_matrix"]]
    mat = df[df["is_matrix"]]

    panels = [
        ("Features vectorielles", vec, [COL_RAW, COL_ARTHUR, COL_POOLED]),
        ("Features matricielles", mat, [COL_RAW, COL_POOLED]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [3, 2]})

    for ax, (title, sub, cols) in zip(axes, panels):
        n_tot = len(sub)
        counts = [int((sub[c] < alpha).sum()) for c in cols]
        x = np.arange(len(cols))
        bars = ax.bar(x, counts, color=[COLORS[c] for c in cols],
                      edgecolor="black", linewidth=0.6, width=0.6)
        for b, n in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    str(n), ha="center", va="bottom", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in cols], fontsize=9)
        ax.set_ylabel(f"tests significatifs (p < {alpha})")
        ax.set_title(f"{title}\n{n_tot} tests au total", fontsize=11)
        ax.set_ylim(0, max(max(counts), 1) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Effet du niveau de correction, schema subject (RFX)",
                 fontsize=13)
    fig.tight_layout()
    out = out_dir / "fig4_n_significatifs_par_niveau.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Ecrit : {out}")


def fig_courbe_survie(df: pd.DataFrame, out_dir: Path, alpha: float):
    """Figure 5 : p triees croissantes par rang, une courbe par niveau.

    IMPORTANT : un panneau par FAMILLE de correction, pas par type de feature.
    Melanger dans une meme courbe des p corrigees sur 95 tests (psd_classic) et
    des p corrigees sur 19 tests (features isolees) produirait un trie qui n'a
    pas d'interpretation unique : les isolees paraitraient plus significatives
    par simple effet de taille de famille.

    Les features isolees ont pooled == arthur par construction (seules dans leur
    famille), les deux courbes se superposent donc exactement. Ce n'est pas un
    bug, c'est la definition.
    """
    df = df.copy()
    df["famille"] = df["feature"].map(assign_family)

    panels = [
        ("psd_classic\n5 bandes x 19 electrodes = 95 tests / etat",
         df[df["famille"] == "psd_classic"], [COL_RAW, COL_ARTHUR, COL_POOLED]),
        ("matrix\n6 features, 1 mesure chacune = 6 tests / etat",
         df[df["famille"] == "matrix"], [COL_RAW, COL_POOLED]),
        ("features isolees\nchacune seule sur ses 19 electrodes",
         df[df["famille"] == "isolee"], [COL_RAW, COL_ARTHUR, COL_POOLED]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, sub, cols) in zip(axes, panels):
        for c in cols:
            vals = np.sort(sub[c].dropna().values)
            if len(vals) == 0:
                continue
            ranks = np.arange(1, len(vals) + 1)
            # pooled trace en pointille sur les isolees : identique a arthur,
            # sinon la courbe bleue est invisible sous la rouge.
            dashed = (c == COL_POOLED and title.startswith("features isolees"))
            ax.plot(ranks, vals, marker="o", markersize=3.5,
                    linewidth=2.2 if dashed else 1.4,
                    linestyle="--" if dashed else "-",
                    alpha=0.75 if dashed else 1.0,
                    color=COLORS[c], label=LABELS[c].replace("\n", " "))
        ax.axhline(alpha, color="black", linestyle="--", linewidth=1.2)
        ax.text(0.98, alpha, f" alpha = {alpha}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("rang du test (p croissantes)")
        ax.set_ylabel("p-value (echelle log)")
        ax.set_title(f"{title}\n{len(sub)} tests affiches", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Courbe de survie des p-values par famille de correction, schema subject (RFX)",
        fontsize=13)
    fig.tight_layout()
    out = out_dir / "fig5_courbe_survie_pvalues.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Ecrit : {out}")


def fig_espace_de_test(df: pd.DataFrame, out_dir: Path):
    """Figure 6 : schema des deux familles de correction.

    Purement illustratif, aucune donnee : montre que maxstat electrode et
    maxstat pooled corrigent deux axes differents et ne sont pas redondants.
    """
    n_vec = int((~df["is_matrix"]).sum() // 4)   # features vectorielles
    n_mat = int(df["is_matrix"].sum() // 4)      # features matricielles

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Bloc gauche : maxstat electrode, une combinaison, 19 electrodes
    ax.add_patch(mpatches.Rectangle((0.4, 1.0), 3.6, 2.8, facecolor="#dce6f0",
                                    edgecolor=COLORS[COL_ARTHUR], linewidth=2))
    ax.text(2.2, 3.5, "maxstat electrode (Arthur)", ha="center",
            fontsize=11, fontweight="bold", color=COLORS[COL_ARTHUR])
    for i in range(19):
        r, c = divmod(i, 7)
        ax.add_patch(mpatches.Rectangle((0.75 + c * 0.44, 2.65 - r * 0.5),
                                        0.34, 0.36, facecolor="white",
                                        edgecolor=COLORS[COL_ARTHUR], linewidth=0.9))
    ax.text(2.2, 1.25, "19 electrodes, pour UNE combinaison\nfeature x stade",
            ha="center", fontsize=9.5)

    # Bloc droite : maxstat pooled, famille de combinaisons
    ax.add_patch(mpatches.Rectangle((5.6, 1.0), 4.0, 2.8, facecolor="#f3dcdc",
                                    edgecolor=COLORS[COL_POOLED], linewidth=2))
    ax.text(7.6, 3.5, "maxstat pooled", ha="center",
            fontsize=11, fontweight="bold", color=COLORS[COL_POOLED])
    for i in range(20):
        r, c = divmod(i, 5)
        ax.add_patch(mpatches.Rectangle((6.05 + c * 0.62, 2.65 - r * 0.5),
                                        0.5, 0.36, facecolor="white",
                                        edgecolor=COLORS[COL_POOLED], linewidth=0.9))
    ax.text(7.6, 1.25,
            f"famille de features x {4} stades\n"
            f"({n_vec} vectorielles, {n_mat} matricielles)",
            ha="center", fontsize=9.5)

    ax.text(4.85, 2.4, "x", ha="center", va="center", fontsize=20,
            color="#666666")
    ax.text(5.0, 0.35,
            "Deux axes de correction distincts, donc non redondants. "
            "Les features matricielles n'ont pas d'axe electrode.",
            ha="center", fontsize=9.5, style="italic", color="#444444")

    out = out_dir / "fig6_espace_de_test.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Ecrit : {out}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_table(args.csv)

    n_vec, n_mat = int((~df["is_matrix"]).sum()), int(df["is_matrix"].sum())
    print(f"Table : {len(df)} lignes ({n_vec} vectorielles, {n_mat} matricielles)")

    fig_n_significatifs(df, args.out_dir, args.alpha)
    fig_courbe_survie(df, args.out_dir, args.alpha)
    fig_espace_de_test(df, args.out_dir)


if __name__ == "__main__":
    main()
