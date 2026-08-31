"""Barplots des accuracies riemanniennes (covariance + cospectres), par stade.

Version "propre" : schema de permutation SUBJECT (RFX, Combrisson et al. 2022).
Deux figures, une par niveau de correction :

  raw    : p non corrigee de chaque feature seule (d["pval"]). Chaque feature a
           sa propre loi nulle ; le seuil (trait par barre) est le quantile
           (1-alpha) de perm_accs de cette feature. Aucune correction inter-features.

  pooled : correction max-stat POOLED sur toute la famille matricielle (6 tests
           par stade). Le trait est un seuil commun au groupe (quantile (1-alpha)
           de la loi nulle du MAXIMUM sur les 6 features). Une barre qui le depasse
           est significative apres correction FWER.

Il n'y a PAS de niveau "max-stat electrodes (Arthur)" ici : une feature matricielle
est un test unique (une matrice par stade), sans dimension electrode a corriger.
Le pendant d'Arthur pour les matrices est le changement de schema (epoch), traite
separement par plot_barplot_riemann_arthur.py.

L'etoile marque p < alpha (raw : p brute ; pooled : p corrigee lue depuis les .npz
de compute_maxstat_correction.py). Trait et etoile sont redondants par construction.

Prerequis pour la figure pooled : compute_maxstat_correction.py en mode pooled sur
la famille 'matrix' :
    python compute_maxstat_correction.py \
        --save-path   .../dream_features_noica_1000hz_overlap \
        --output-path .../dream_features_noica_1000hz_overlap_corrected \
        --family-name matrix \
        --keys cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta

La figure raw ne depend que des results/*.npz (pval, perm_accs), pas du corrected.

Usage :
    python plot_barplot_riemann_clean.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /home/alouis/dream-recall-alex/plot_overlap \
        --alpha 0.05
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans affichage, obligatoire sur noeud de calcul
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    MATRIX_KEYS,
    RESOLUTION,
    STATES_ORDERED,
    band_label,
    key_color,
    load_maxstat,
    load_null_max,
    load_result,
    maxstat_threshold,
)

WIDTH = 0.90
Y_LABEL = "Decoding accuracy (%)"

LEVELS = {
    "raw":    "non corrige (p brute, feature seule)",
    "pooled": "max-stat pooled sur la famille matricielle",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, contenant results/.")
    p.add_argument("--corrected-path", type=Path, required=True,
                   help="Dossier des .npz maxstat (compute_maxstat_correction.py).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Dossier de sortie des figures.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Seuil de significativite.")
    p.add_argument("--ymin", type=float, default=None,
                   help="Borne basse de l'axe y (%%). Defaut : calculee sur les donnees.")
    p.add_argument("--ymax", type=float, default=None,
                   help="Borne haute de l'axe y (%%).")
    return p.parse_args()


def collect(save_path: Path, corrected_path: Path, level: str, alpha: float):
    """Charge accuracies, ecarts-types, decisions et seuils pour tous les combos.

    Retourne (accs, stds, sigs, bar_thr, group_thr) indexes [stade][feature],
    sauf group_thr indexe [stade] (seuil commun, pooled seulement).

      raw    : bar_thr[stade][feat] = quantile (1-alpha) de perm_accs de la feature.
               group_thr = NaN (pas de seuil commun).
      pooled : group_thr[stade] = quantile (1-alpha) de la nulle du max.
               bar_thr = NaN (le trait est groupe, pas par barre).
    """
    accs, stds, sigs, bar_thr, group_thr = [], [], [], [], []
    for state in STATES_ORDERED:
        if level == "pooled":
            pvals = load_maxstat(corrected_path, "matrix", state)
            null_max = load_null_max(corrected_path, "matrix", state)
            group_thr.append(
                maxstat_threshold(null_max, alpha) * 100 if null_max is not None else np.nan
            )
        else:
            pvals = None
            group_thr.append(np.nan)

        a_row, s_row, sig_row, t_row = [], [], [], []
        for key in MATRIX_KEYS:
            d = load_result(save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                a_row.append(np.nan); s_row.append(np.nan)
                sig_row.append(False); t_row.append(np.nan)
                continue

            a_row.append(float(d["acc_mean"]) * 100)
            # acc_std est deja la dispersion inter-bootstrap.
            s_row.append(float(d["acc_std"]) * 100)

            if level == "raw":
                p = float(d["pval"])
                # seuil par barre : quantile (1-alpha) de la nulle de la feature
                null = np.array(d["perm_accs"], dtype=float)
                t_row.append(maxstat_threshold(null, alpha) * 100)
            else:  # pooled
                p = pvals.get(key, 1.0) if pvals is not None else 1.0
                t_row.append(np.nan)

            sig_row.append(p < alpha)

        accs.append(a_row); stds.append(s_row); sigs.append(sig_row)
        bar_thr.append(t_row)
    return accs, stds, sigs, bar_thr, group_thr


def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax):
    accs, stds, sigs, bar_thr, group_thr = collect(save_path, corrected_path, level, alpha)

    fig, ax = plt.subplots(figsize=(11, 5))
    n_keys = len(MATRIX_KEYS)
    group_width = n_keys + 1  # une barre vide entre les groupes de stades

    handles = []
    n_sig_total = 0
    for g, state in enumerate(STATES_ORDERED):
        for i, key in enumerate(MATRIX_KEYS):
            val = accs[g][i]
            if np.isnan(val):
                continue
            x = g * group_width + i
            b = ax.bar(x, val, WIDTH, color=key_color(key), yerr=stds[g][i],
                       capsize=2, error_kw=dict(lw=1))
            if g == 0:
                handles.append(b)

            if sigs[g][i]:
                n_sig_total += 1
                ax.text(x, val + stds[g][i] + 0.4, "*", ha="center", va="bottom",
                        fontsize=15, fontweight="bold")

            # raw : trait par barre (seuil propre a la feature)
            if level == "raw" and not np.isnan(bar_thr[g][i]):
                t = bar_thr[g][i]
                ax.plot([x - WIDTH / 2, x + WIDTH / 2], [t, t], "k--", lw=1.2, zorder=3)

        # pooled : un seul trait par groupe (seuil commun a la famille)
        if level == "pooled" and not np.isnan(group_thr[g]):
            x0 = g * group_width - WIDTH / 2
            x1 = g * group_width + (n_keys - 1) + WIDTH / 2
            ax.plot([x0, x1], [group_thr[g], group_thr[g]], "k--", lw=1.2, zorder=3)

    finite = [v for row in accs for v in row if not np.isnan(v)]
    lo = ymin if ymin is not None else min(48, min(finite) - 3)
    hi = ymax if ymax is not None else max(finite) + 5
    ax.set_ylim(lo, hi)

    ax.set_ylabel(Y_LABEL)
    ax.set_title(
        f"Riemannian classifications, permutation sujet (RFX), "
        f"{LEVELS[level]}, p < {alpha}"
    )
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATES_ORDERED))])
    ax.set_xticklabels(STATES_ORDERED)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)  # niveau de chance
    ax.spines[["top", "right"]].set_visible(False)

    if handles:
        ax.legend(handles, [band_label(k) for k in MATRIX_KEYS],
                  frameon=False, fontsize=9, ncol=2,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))

    if level == "pooled":
        note = "- - -  seuil max-stat pooled   |   *  p < %.2g corrige" % alpha
    else:
        note = "- - -  seuil par feature   |   *  p < %.2g non corrige" % alpha
    ax.text(0.995, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.3")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"barplot_riemann_subject_{level}_p{alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"  [{level}] {n_sig_total} barres significatives sur "
          f"{n_keys * len(STATES_ORDERED)} -> {out}")
    return out


def main() -> None:
    args = parse_args()
    print(f"=== barplots riemanniens (subject/RFX, p < {args.alpha}) ===")
    for level in ("raw", "pooled"):
        make_figure(args.save_path, args.corrected_path, args.out_dir,
                    level, args.alpha, args.ymin, args.ymax)


if __name__ == "__main__":
    main()
