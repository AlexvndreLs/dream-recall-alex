"""Barplot des features vectorielles significatives (branche noica_1000hz_overlap).

Lit pvalue_summary_table.csv (produit par build_pvalue_summary_table.py) et trace
un barplot des features vectorielles retenues : hauteur = accuracy (%), ligne de
chance a 50%, et au-dessus de chaque barre trois marqueurs empiles indiquant la
significativite sous chacune des trois corrections (schema subject/RFX) :

    raw    = p_non_corrige_subject       (aucune correction)
    arthur = p_maxstat_arthur_subject    (maxstat sur les 19 electrodes)
    pooled = p_maxstat_pooled_subject    (maxstat pooled famille PSD)

Un marqueur n'est affiche que si la p correspondante est < ALPHA (defaut 0.05).

Usage typique (sur Fir) :
    python3 plot_barplot_vector_sig.py \
        --csv /scratch/alouis/dream_features_noica_1000hz_overlap/results/pvalue_summary_table.csv \
        --out plot_overlap/barplot_vector_sig.png

Par defaut, les features tracees sont les trois retenues comme significatives sous
pooled (psd_sigma/S2, psd_osc_beta/SWS, psd_osc_delta/SWS). Modifiable via --features.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.05

# (feature, state) affiches par defaut, dans cet ordre
DEFAULT_FEATURES = [
    ("psd_sigma", "S2"),
    ("psd_osc_beta", "SWS"),
    ("psd_osc_delta", "SWS"),
]

# colonnes du CSV utilisees (schema subject uniquement)
COL_ACC = "accuracy_pct"
COL_RAW = "p_non_corrige_subject"
COL_ARTHUR = "p_maxstat_arthur_subject"
COL_POOLED = "p_maxstat_pooled_subject"

# marqueurs empiles : (label affiche, colonne p, couleur)
CORRECTIONS = [
    ("raw", COL_RAW, "#9e9e9e"),
    ("arthur", COL_ARTHUR, "#f9a825"),
    ("pooled", COL_POOLED, "#2e7d32"),
]


def _parse_features(spec):
    """'psd_sigma/S2,psd_osc_beta/SWS' -> [('psd_sigma','S2'), ...]."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "/" not in tok:
            raise ValueError(f"format attendu feature/state, recu : {tok!r}")
        feat, state = tok.split("/", 1)
        out.append((feat.strip(), state.strip()))
    return out


def load_rows(csv_path, wanted):
    """Retourne, pour chaque (feature,state) demande, un dict des champs utiles.

    Echoue explicitement si une combinaison demandee est absente ou si une p-value
    attendue vaut PENDING (pas de repli silencieux)."""
    index = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            index[(r["feature"], r["state"])] = r

    rows = []
    for key in wanted:
        if key not in index:
            raise KeyError(
                f"combinaison {key[0]}/{key[1]} absente de {csv_path}"
            )
        r = index[key]
        acc = float(r[COL_ACC])
        ps = {}
        for label, col, _ in CORRECTIONS:
            val = r[col]
            if val == "PENDING":
                raise ValueError(
                    f"{key[0]}/{key[1]} : colonne {col} = PENDING "
                    f"(perms non calculees pour cette correction)"
                )
            ps[label] = float(val)
        rows.append({"feature": key[0], "state": key[1], "acc": acc, "p": ps})
    return rows


def plot(rows, out_path, alpha):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * n + 2), 6))

    x = list(range(n))
    accs = [r["acc"] for r in rows]
    bars = ax.bar(x, accs, width=0.55, color="#5c6bc0", edgecolor="black", zorder=3)

    # ligne de chance
    ax.axhline(50, ls="--", lw=1, color="black", zorder=2)
    ax.text(n - 0.5, 50.4, "chance (50%)", ha="right", va="bottom", fontsize=9)

    # marqueurs empiles au-dessus de chaque barre
    for xi, r in zip(x, rows):
        top = r["acc"]
        step = 1.1  # espacement vertical entre marqueurs (en % d'accuracy)
        for j, (label, _, color) in enumerate(CORRECTIONS):
            p = r["p"][label]
            y = top + 0.8 + j * step
            if p < alpha:
                ax.text(
                    xi, y, f"* {label}", ha="center", va="bottom",
                    fontsize=10, color=color, fontweight="bold",
                )
            else:
                ax.text(
                    xi, y, f"n.s. {label}", ha="center", va="bottom",
                    fontsize=8, color="#cccccc",
                )

    # valeur d'accuracy dans la barre
    for b, a in zip(bars, accs):
        ax.text(
            b.get_x() + b.get_width() / 2, a - 2.5, f"{a:.1f}",
            ha="center", va="top", fontsize=9, color="white", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['feature']}\n{r['state']}" for r in rows], fontsize=10
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, max(accs) + 6)
    ax.set_title(
        "Features vectorielles significatives (noica 1000Hz overlap, subject/RFX)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legende des corrections
    handles = [
        plt.Line2D([0], [0], marker="*", ls="", color=c, label=lab, markersize=10)
        for lab, _, c in CORRECTIONS
    ]
    ax.legend(
        handles=handles, title=f"significatif si p < {alpha}",
        loc="upper right", fontsize=9, framealpha=0.9,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"figure ecrite : {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="chemin du pvalue_summary_table.csv")
    ap.add_argument(
        "--out", default="plot_overlap/barplot_vector_sig.png",
        help="chemin de sortie du PNG",
    )
    ap.add_argument(
        "--features", default=None,
        help="liste feature/state separee par des virgules "
             "(defaut : les 3 significatives)",
    )
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    wanted = _parse_features(args.features) if args.features else DEFAULT_FEATURES
    rows = load_rows(args.csv, wanted)
    for r in rows:
        flags = " ".join(
            f"{lab}={r['p'][lab]:.4f}{'*' if r['p'][lab] < args.alpha else ''}"
            for lab, _, _ in CORRECTIONS
        )
        print(f"  {r['feature']}/{r['state']}: acc={r['acc']:.2f}  {flags}")
    plot(rows, args.out, args.alpha)


if __name__ == "__main__":
    main()
