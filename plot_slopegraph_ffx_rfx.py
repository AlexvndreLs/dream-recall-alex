"""Option B, slopegraph des p-values : permutation epoque (FFX) vers sujet (RFX).

Deux colonnes verticales : a gauche la p-value sous permutation epoque
(Fixed-effects), a droite sous permutation sujet (Random-effects). Une ligne par
combinaison feature x stade relie ses deux p-values. Axe y en log (p-value
croissante vers le haut).

Toutes les lignes montent de gauche a droite : la permutation sujet donne
systematiquement des p-values plus grandes (plus conservatrices). Le seuil
alpha = 0.05 est trace en pointilles ; on voit combien de combos passent de
significatif (sous la ligne a gauche) a non significatif (au-dessus a droite).

Les lignes qui restent significatives des deux cotes (encore sous 0.05 a droite)
sont mises en evidence en couleur, les autres en gris.

Usage (sur Fir) :
    python3 plot_slopegraph_ffx_rfx.py \
        --results /scratch/alouis/dream_features_noica_1000hz/results \
        --out plot_perm_explication/slopegraph_ffx_rfx.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUFFIX = "_epochperm.npz"
ALPHA = 0.05


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
    p.add_argument("--out", default="plot_perm_explication/slopegraph_ffx_rfx.png")
    p.add_argument("--floor", type=float, default=None)
    p.add_argument("--label-survivors", action="store_true",
                   help="Ecrire le nom des combos encore significatifs a droite.")
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

    x_ffx, x_rfx = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6.5, 7))

    n_sig_ffx = int(np.sum(p_ffx < ALPHA))
    n_sig_rfx = int(np.sum(p_rfx < ALPHA))
    n_lost = int(np.sum((p_ffx < ALPHA) & (p_rfx >= ALPHA)))

    for i, combo in enumerate(combos):
        survives = p_rfx[i] < ALPHA
        color = "#c0392b" if survives else "#bcbcbc"
        lw = 1.6 if survives else 0.8
        alpha = 0.95 if survives else 0.5
        z = 3 if survives else 1
        ax.plot([x_ffx, x_rfx], [p_ffx[i], p_rfx[i]],
                "-", color=color, linewidth=lw, alpha=alpha, zorder=z)
        ax.plot([x_ffx, x_rfx], [p_ffx[i], p_rfx[i]],
                "o", color=color, markersize=4, alpha=alpha, zorder=z)
        if args.label_survivors and survives:
            ax.text(x_rfx + 0.03, p_rfx[i], combo, fontsize=7,
                    va="center", ha="left", color="#c0392b")

    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(1.15, ALPHA, "alpha = 0.05", fontsize=9, va="center", ha="left")

    ax.set_yscale("log")
    ax.set_xlim(-0.25, 1.55 if args.label_survivors else 1.35)
    ax.set_xticks([x_ffx, x_rfx])
    ax.set_xticklabels(
        ["permutation\nepoque\n(Fixed-effects)", "permutation\nsujet\n(Random-effects)"],
        fontsize=10,
    )
    ax.set_ylabel("p-value (echelle log)", fontsize=11)
    ax.set_title(
        "Effet du schema de permutation sur la significativite\n"
        f"significatifs : {n_sig_ffx} (epoque) -> {n_sig_rfx} (sujet), "
        f"{n_lost} perdus",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[OK] figure ecrite : {args.out}")
    print(f"[info] {len(combos)} combos | sig epoque={n_sig_ffx} "
          f"sig sujet={n_sig_rfx} perdus={n_lost}")


if __name__ == "__main__":
    main()