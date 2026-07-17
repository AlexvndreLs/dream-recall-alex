"""Barplot des accuracies riemanniennes (covariance + cospectres), par stade.

Version "propre" : schéma de permutation SUBJECT (RFX, Combrisson et al. 2022),
correction max-stat POOLED sur toute la famille matricielle (6 tests par stade).
À distinguer de plot_barplot_riemann_arthur.py, qui réplique le chapitre 1 avec
le schéma epoch et sans correction inter-features.

Deux marquages superposés, qui ne disent pas la même chose :
  - le trait pointillé est le seuil corrigé, quantile (1-alpha) de la loi nulle
    du MAXIMUM sur les 6 features de la famille. Une barre qui le dépasse est
    significative après correction FWER.
  - l'étoile marque les barres dont p_corrigé < alpha, lu depuis les .npz de
    compute_maxstat_correction.py. Trait et étoile sont redondants par
    construction : l'étoile rend la décision lisible sans mesurer à l'oeil.

Prérequis : compute_maxstat_correction.py doit avoir tourné en mode pooled sur
la famille 'matrix' :
    python compute_maxstat_correction.py \
        --save-path   .../dream_features_noica_1000hz_overlap \
        --output-path .../dream_features_noica_1000hz_overlap_corrected \
        --family-name matrix \
        --keys cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta

Sans ces fichiers, la figure est tracée sans seuil ni étoile plutôt que
d'échouer : les accuracies restent lisibles, seule la décision statistique
manque.

Usage :
    python plot_barplot_riemann_clean.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /scratch/alouis/dream-recall-alex/plot_overlap \
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, contenant results/.")
    p.add_argument("--corrected-path", type=Path, required=True,
                   help="Dossier des .npz maxstat (compute_maxstat_correction.py).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Dossier de sortie de la figure.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Seuil de significativité (max-stat pooled).")
    p.add_argument("--ymin", type=float, default=None,
                   help="Borne basse de l'axe y (%%). Défaut : calculée sur les données.")
    p.add_argument("--ymax", type=float, default=None,
                   help="Borne haute de l'axe y (%%).")
    return p.parse_args()


def collect(save_path: Path, corrected_path: Path, alpha: float):
    """Charge accuracies, écarts-types, seuils et décisions pour tous les combos.

    Retourne (accs, stds, sigs, thresholds) où accs/stds/sigs sont indexés
    [stade][feature] et thresholds [stade] (un seul seuil par stade, commun à
    toute la famille : c'est le principe du pooling).
    """
    accs, stds, sigs, thresholds = [], [], [], []
    for state in STATES_ORDERED:
        pvals = load_maxstat(corrected_path, "matrix", state)
        null_max = load_null_max(corrected_path, "matrix", state)
        thresholds.append(
            maxstat_threshold(null_max, alpha) * 100 if null_max is not None else np.nan
        )

        a_row, s_row, sig_row = [], [], []
        for key in MATRIX_KEYS:
            d = load_result(save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                a_row.append(np.nan)
                s_row.append(np.nan)
                sig_row.append(False)
                continue
            a_row.append(float(d["acc_mean"]) * 100)
            # acc_std est déjà la dispersion inter-bootstrap (chaque acc_scores
            # est une accuracy moyennée sur les 324 splits par run_cv) : pas de
            # regroupement supplémentaire à faire.
            s_row.append(float(d["acc_std"]) * 100)
            sig_row.append(pvals is not None and pvals.get(key, 1.0) < alpha)

        accs.append(a_row)
        stds.append(s_row)
        sigs.append(sig_row)
    return accs, stds, sigs, thresholds


def main() -> None:
    args = parse_args()
    print(f"=== barplot riemannien (subject, max-stat pooled, p < {args.alpha}) ===")

    accs, stds, sigs, thresholds = collect(args.save_path, args.corrected_path, args.alpha)

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
                # étoile posée au-dessus de la barre d'erreur, pas de la barre
                ax.text(x, val + stds[g][i] + 0.4, "*", ha="center", va="bottom",
                        fontsize=15, fontweight="bold")

        # Un seul trait par groupe : le seuil est commun à la famille entière.
        t = thresholds[g]
        if not np.isnan(t):
            x0 = g * group_width - WIDTH / 2
            x1 = g * group_width + (n_keys - 1) + WIDTH / 2
            ax.plot([x0, x1], [t, t], "k--", lw=1.2, zorder=3)

    # Bornes : laisse respirer au-dessus des étoiles, coupe sous le niveau de
    # chance seulement si les données descendent plus bas.
    finite = [v for row in accs for v in row if not np.isnan(v)]
    ymin = args.ymin if args.ymin is not None else min(48, min(finite) - 3)
    ymax = args.ymax if args.ymax is not None else max(finite) + 5
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel(Y_LABEL)
    ax.set_title(
        f"Riemannian classifications — permutation sujet (RFX), "
        f"max-stat pooled sur {n_keys} features, p < {args.alpha}"
    )
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATES_ORDERED))])
    ax.set_xticklabels(STATES_ORDERED)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)  # niveau de chance
    ax.spines[["top", "right"]].set_visible(False)

    if handles:
        ax.legend(handles, [band_label(k) for k in MATRIX_KEYS],
                  frameon=False, fontsize=9, ncol=2)

    # Légende du trait et de l'étoile : sans ça, le lecteur ne peut pas savoir
    # que les deux disent la même chose ni sur quoi porte la correction.
    ax.text(0.995, 0.02,
            "- - -  seuil max-stat pooled   |   *  p < %.2g corrigé" % args.alpha,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.3")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"barplot_riemann_subject_pooled_p{args.alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"  {n_sig_total} barres significatives sur {n_keys * len(STATES_ORDERED)}")
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
