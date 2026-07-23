"""Scatter des p-values Fixed-effects (epoque) vs Random-effects (sujet).

Pour chaque combinaison feature x stade, deux fichiers .npz coexistent dans le
dossier results/ :

    {combo}.npz            -> permutation au niveau du sujet   (Random-effects, RFX)
    {combo}_epochperm.npz  -> permutation au niveau de l'epoque (Fixed-effects, FFX)

Les deux contiennent la cle 'pvals' (un tableau de p-values, une par electrode pour
les features vectorielles). Ce script apparie les fichiers, prend pour chaque combo
la p-value minimale sur les electrodes (la plus favorable, celle qui pilote la
significativite affichee), et trace :

    x = p-value Fixed-effects (epoque)
    y = p-value Random-effects (sujet)

en echelle log-log, avec la ligne d'identite y = x. Tous les points tombent
au-dessus de la diagonale (RFX plus conservateur que FFX). Le facteur median
p_RFX / p_FFX est annote sur la figure : il quantifie de combien la permutation
epoque degonfle les p-values.

Usage (sur Fir) :
    python3 plot_pvalue_scatter_ffx_rfx.py \
        --results /scratch/alouis/dream_features_noica_1000hz/results \
        --out plot_perm_explication/pvalue_scatter_ffx_rfx.png

Note : ne considere que les combos ou les DEUX fichiers existent.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUFFIX = "_epochperm.npz"


def load_min_pval(path):
    """Retourne la p-value minimale sur les electrodes pour un .npz, ou None."""
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[skip] {os.path.basename(path)} : {e}")
        return None
    if "pvals" not in d:
        print(f"[skip] {os.path.basename(path)} : pas de cle 'pvals'")
        return None
    pv = np.asarray(d["pvals"], dtype=float).ravel()
    if pv.size == 0:
        return None
    return float(np.min(pv))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        default="/scratch/alouis/dream_features_noica_1000hz/results",
        help="Dossier contenant les paires {combo}.npz et {combo}_epochperm.npz",
    )
    p.add_argument(
        "--out",
        default="plot_perm_explication/pvalue_scatter_ffx_rfx.png",
        help="Chemin du PNG de sortie",
    )
    p.add_argument(
        "--floor",
        type=float,
        default=None,
        help="Plancher applique aux p-values nulles (defaut : 1/(n_perm+1) devine, "
        "sinon 1e-4). Evite log(0).",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Trouver toutes les paires
    epoch_files = sorted(glob.glob(os.path.join(args.results, "*" + SUFFIX)))
    if not epoch_files:
        raise SystemExit(
            f"Aucun fichier '*{SUFFIX}' dans {args.results}. "
            "Verifie le chemin, ou que la permutation epoque a bien ete calculee."
        )

    combos = []
    p_ffx = []
    p_rfx = []
    for ef in epoch_files:
        combo = os.path.basename(ef)[: -len(SUFFIX)]
        rf = os.path.join(args.results, combo + ".npz")
        if not os.path.exists(rf):
            print(f"[skip] {combo} : pas de fichier RFX apparie")
            continue
        pf = load_min_pval(ef)
        pr = load_min_pval(rf)
        if pf is None or pr is None:
            continue
        combos.append(combo)
        p_ffx.append(pf)
        p_rfx.append(pr)

    if not combos:
        raise SystemExit("Aucune paire exploitable trouvee.")

    p_ffx = np.asarray(p_ffx, dtype=float)
    p_rfx = np.asarray(p_rfx, dtype=float)

    # Plancher pour eviter log(0). Si non fourni, on devine 1/(n_perm+1) a partir
    # de la plus petite p non nulle observee, sinon 1e-4.
    if args.floor is not None:
        floor = args.floor
    else:
        nonzero = np.concatenate([p_ffx[p_ffx > 0], p_rfx[p_rfx > 0]])
        floor = float(nonzero.min()) if nonzero.size else 1e-4
    p_ffx = np.clip(p_ffx, floor, 1.0)
    p_rfx = np.clip(p_rfx, floor, 1.0)

    # Facteur median de deflation
    ratio = p_rfx / p_ffx
    med_ratio = float(np.median(ratio))

    fig, ax = plt.subplots(figsize=(6.5, 6.2))

    ax.scatter(
        p_ffx,
        p_rfx,
        s=42,
        c="#c0392b",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    # Ligne d'identite
    lo = floor * 0.7
    hi = 1.2
    ax.plot([lo, hi], [lo, hi], "--", color="#444444", linewidth=1.3, zorder=2,
            label="identite (y = x)")

    # Ligne du seuil alpha = 0.05 sur les deux axes
    ax.axhline(0.05, color="#2c7fb8", linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(0.05, color="#2c7fb8", linewidth=1.0, linestyle=":", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel(
        "p-value, permutation au niveau de l'epoque (Fixed-effects)", fontsize=11
    )
    ax.set_ylabel(
        "p-value, permutation au niveau du sujet (Random-effects)", fontsize=11
    )
    ax.set_title(
        "Deflation des p-values par la permutation epoque\n"
        f"n = {len(combos)} combinaisons, facteur median p_sujet / p_epoque "
        f"= {med_ratio:.0f}x",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(True, which="both", linestyle="-", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[OK] figure ecrite : {args.out}")
    print(f"[info] {len(combos)} combos, facteur median = {med_ratio:.1f}x")


if __name__ == "__main__":
    main()