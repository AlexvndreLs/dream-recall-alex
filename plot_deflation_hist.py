"""Option A, histogramme du facteur de deflation des p-values (RFX / FFX).

Pour chaque combinaison feature x stade, deux fichiers .npz coexistent dans
results/ :

    {combo}.npz            -> permutation au niveau du sujet   (Random-effects, RFX)
    {combo}_epochperm.npz  -> permutation au niveau de l'epoque (Fixed-effects, FFX)

On prend pour chaque combo la p-value minimale sur les electrodes (celle qui pilote
la significativite affichee) dans chaque schema, puis le facteur de deflation :

    facteur = p_RFX / p_FFX  (toujours >= 1 : la permutation epoque sous-estime la p)

La figure trace la distribution de ce facteur en echelle log, avec la mediane
annotee. Message : dans la grande majorite des combos, la permutation epoque
degonfle la p-value d'un facteur de plusieurs dizaines.

Usage (sur Fir) :
    python3 plot_deflation_hist.py \
        --results /scratch/alouis/dream_features_noica_1000hz/results \
        --out plot_perm_explication/deflation_hist.png
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
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[skip] {os.path.basename(path)} : {e}")
        return None
    if "pvals" not in d:
        return None
    pv = np.asarray(d["pvals"], dtype=float).ravel()
    if pv.size == 0:
        return None
    return float(np.min(pv))


def collect_pairs(results):
    epoch_files = sorted(glob.glob(os.path.join(results, "*" + SUFFIX)))
    combos, p_ffx, p_rfx = [], [], []
    for ef in epoch_files:
        combo = os.path.basename(ef)[: -len(SUFFIX)]
        rf = os.path.join(results, combo + ".npz")
        if not os.path.exists(rf):
            continue
        pf, pr = load_min_pval(ef), load_min_pval(rf)
        if pf is None or pr is None:
            continue
        combos.append(combo)
        p_ffx.append(pf)
        p_rfx.append(pr)
    return combos, np.asarray(p_ffx), np.asarray(p_rfx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/scratch/alouis/dream_features_noica_1000hz/results")
    p.add_argument("--out", default="plot_perm_explication/deflation_hist.png")
    p.add_argument("--floor", type=float, default=None,
                   help="Plancher applique aux p nulles (defaut : plus petite p non nulle observee).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    combos, p_ffx, p_rfx = collect_pairs(args.results)
    if not combos:
        raise SystemExit(f"Aucune paire exploitable dans {args.results}.")

    if args.floor is not None:
        floor = args.floor
    else:
        nz = np.concatenate([p_ffx[p_ffx > 0], p_rfx[p_rfx > 0]])
        floor = float(nz.min()) if nz.size else 1e-4
    p_ffx = np.clip(p_ffx, floor, 1.0)
    p_rfx = np.clip(p_rfx, floor, 1.0)

    ratio = p_rfx / p_ffx
    med = float(np.median(ratio))

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # bins log
    lo = max(0.5, ratio.min() * 0.8)
    hi = ratio.max() * 1.2
    bins = np.logspace(np.log10(lo), np.log10(hi), 24)
    ax.hist(ratio, bins=bins, color="#c0392b", alpha=0.75, edgecolor="white")

    ax.axvline(med, color="black", linestyle="--", linewidth=1.6,
               label=f"mediane = {med:.0f}x")
    ax.axvline(1.0, color="#2c7fb8", linestyle=":", linewidth=1.3,
               label="pas de deflation (facteur 1)")

    ax.set_xscale("log")
    ax.set_xlabel("facteur de deflation  p(sujet) / p(epoque)", fontsize=11)
    ax.set_ylabel("nombre de combinaisons feature x stade", fontsize=11)
    ax.set_title(
        "De combien la permutation epoque degonfle les p-values\n"
        f"n = {len(combos)} combinaisons",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[OK] figure ecrite : {args.out}")
    print(f"[info] {len(combos)} combos, mediane facteur = {med:.1f}x, "
          f"min {ratio.min():.1f}x, max {ratio.max():.1f}x")


if __name__ == "__main__":
    main()