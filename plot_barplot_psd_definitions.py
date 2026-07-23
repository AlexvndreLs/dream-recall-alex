"""Comparaison des trois definitions de la puissance spectrale, par bande et stade.

Trois definitions tracees cote a cote pour chaque bande :

  psd_{band}      puissance brute Welch, V^2/Hz. Aucune separation aperiodic.
  psd_osc_{band}  RATIO P / A         (feat_extract_umap_fooof_v4.fit_fooof)
  psd_sub_{band}  SOUSTRACTION P - A  (feat_extract_sub.fit_fooof_sub)

ou A = 10 ** ap_fit_log est le fit aperiodic FOOOF reconstruit en lineaire.

Pourquoi cette figure
---------------------
Le diagnostic du 23/07 (job diag_sub) a montre que ratio et soustraction ne sont
PAS deux echelles de la meme quantite : correlation par bande r=0.39-0.45 en
delta, 0.72-0.84 en beta, 0.90-0.93 en sigma, et la soustraction produit une
majorite de valeurs negatives en delta (49-74%) la ou le ratio reste positif
autour de 1. Le choix de definition est donc un choix methodologique qui peut
changer les conclusions, pas une convention d'affichage.

Cette figure rend ce choix visible : une bande dont les trois barres sont
comparables est robuste a la definition ; une bande dont seule une barre passe
le seuil depend de la normalisation par le 1/f.

Hauteur de barre = accuracy de la MEILLEURE electrode parmi 19 (meme convention
que plot_barplot_vector_clean.py et que la Fig. 4 d'Arthur). Error bar =
acc_std inter-bootstrap de cette electrode.

Correction
----------
Trois niveaux, une figure chacun, l'etoile marque p < alpha :

  raw     p non corrigee de la meilleure electrode. Le plus permissif.
  arthur  max-stat sur les 19 electrodes, feature seule
          (compute_maxstat_correction.py --mode arthur).
  pooled  max-stat sur le pool des 5 bandes de la famille, 95 tests
          (--mode pooled, family-name psd_classic / psd_osc / psd_sub).

En pooled, chaque famille a SON pool, donc son seuil : trois traits pointilles
distincts par groupe de stade. C'est voulu, les trois familles sont trois
analyses separees, pas un pool commun de 285 tests.

Prerequis
---------
    SRC=/scratch/alouis/dream_features_noica_1000hz_overlap
    SUB=/scratch/alouis/dream_features_noica_1000hz_sub
    python compute_maxstat_correction.py --save-path $SUB \
        --output-path ${SUB}_corrected --family-name psd_sub --mode pooled \
        --keys psd_sub_delta psd_sub_theta psd_sub_alpha psd_sub_sigma psd_sub_beta
    # psd_classic et psd_osc pooled existent deja dans ${SRC}_corrected

Usage
-----
    python plot_barplot_psd_definitions.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --sub-path       /scratch/alouis/dream_features_noica_1000hz_sub \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --sub-corrected-path /scratch/alouis/dream_features_noica_1000hz_sub_corrected \
        --out-dir        /home/alouis/dream-recall-alex/plot_overlap \
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
    BAND_COLORS,
    RESOLUTION,
    STATES_ORDERED,
    load_result,
    maxstat_threshold,
)

# Les trois definitions, dans l'ordre d'affichage a l'interieur d'une bande.
# family : nom passe a --family-name lors du pooling (pour retrouver le .npz).
DEFS = [
    dict(prefix="psd_",     family="psd_classic", label="brute",       hatch=""),
    dict(prefix="psd_osc_", family="psd_osc",     label="ratio",       hatch="//"),
    dict(prefix="psd_sub_", family="psd_sub",     label="soustraction", hatch="xx"),
]

WIDTH = 0.26          # largeur d'une barre, 3 barres tiennent dans 1.0
BAND_STEP = 1.0       # pas entre deux bandes
GROUP_GAP = 1.0       # espace entre deux stades

LEVELS = {
    "raw":    "non corrige (p brute, meilleure electrode)",
    "arthur": "max-stat 19 electrodes (feature seule)",
    "pooled": "max-stat pooled (5 bandes par famille, 95 tests)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Branche overlap : psd_* et psd_osc_*.")
    p.add_argument("--sub-path", type=Path, required=True,
                   help="Branche sub : psd_sub_*.")
    p.add_argument("--corrected-path", type=Path, required=True,
                   help="Maxstat de la branche overlap.")
    p.add_argument("--sub-corrected-path", type=Path, required=True,
                   help="Maxstat de la branche sub.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    return p.parse_args()


def paths_for(d: dict, args) -> tuple[Path, Path]:
    """(save_path, corrected_path) selon la definition : sub vit ailleurs."""
    if d["prefix"] == "psd_sub_":
        return args.sub_path, args.sub_corrected_path
    return args.save_path, args.corrected_path


def arthur_pval(corrected: Path, key: str, state: str, best: int):
    f = corrected / f"{key}_{state}_maxstat_arthur.npz"
    if not f.exists():
        return None
    return float(np.load(f, allow_pickle=True)["pvals_corrected"][best])


def pooled_pval(corrected: Path, family: str, key: str, state: str):
    """p pooled minimale parmi les electrodes de cette key.

    Les test_labels du .npz pooled sont de la forme 'psd_osc_beta/O1'.
    """
    f = corrected / f"{family}_{state}_maxstat.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    labels = [str(x) for x in d["test_labels"]]
    pv = np.array(d["pvals_corrected"], dtype=float)
    mask = np.array([lab.startswith(f"{key}/") for lab in labels])
    return float(pv[mask].min()) if mask.any() else None


def pooled_threshold(corrected: Path, family: str, state: str, alpha: float):
    f = corrected / f"{family}_{state}_maxstat.npz"
    if not f.exists():
        return np.nan
    return maxstat_threshold(np.load(f, allow_pickle=True)["null_max"], alpha) * 100


def bar_threshold(save, corrected, key, state, best, level, alpha):
    """Seuil d'accuracy (%) propre a une barre, pour les niveaux raw et arthur."""
    if level == "raw":
        d = load_result(save, key, state)
        if d is None or "perm_accs" not in d:
            return np.nan
        return maxstat_threshold(np.array(d["perm_accs"])[:, best], alpha) * 100
    if level == "arthur":
        f = corrected / f"{key}_{state}_maxstat_arthur.npz"
        if not f.exists():
            return np.nan
        return maxstat_threshold(np.load(f, allow_pickle=True)["null_max"], alpha) * 100
    return np.nan


def collect(args, level: str):
    """cells[(state, band, def_idx)] -> dict(acc, std, sig, thr, elec).

    Une entree manquante est simplement absente du dict : l'appelant saute la
    barre plutot que de faire echouer la figure entiere.
    """
    cells, pool_thr = {}, {}
    for state in STATES_ORDERED:
        for di, d in enumerate(DEFS):
            save, corrected = paths_for(d, args)
            if level == "pooled":
                pool_thr[(state, di)] = pooled_threshold(
                    corrected, d["family"], state, args.alpha)
            for band in BANDS:
                key = d["prefix"] + band
                res = load_result(save, key, state)
                if res is None:
                    print(f"  absent : {key}_{state}.npz")
                    continue
                acc = np.array(res["acc_mean"], dtype=float)
                std = np.array(res["acc_std"], dtype=float)
                best = int(np.argmax(acc))
                ch = res["ch_names"].tolist() if "ch_names" in res else list(range(len(acc)))

                if level == "raw":
                    p = float(np.array(res["pvals"], dtype=float)[best])
                elif level == "arthur":
                    p = arthur_pval(corrected, key, state, best)
                else:
                    p = pooled_pval(corrected, d["family"], key, state)

                cells[(state, band, di)] = dict(
                    acc=acc[best] * 100,
                    std=float(std[best]) * 100,
                    sig=(p is not None and p < args.alpha),
                    thr=bar_threshold(save, corrected, key, state, best, level, args.alpha),
                    elec=str(ch[best]),
                )
    return cells, pool_thr


def make_figure(args, level: str):
    cells, pool_thr = collect(args, level)
    if not cells:
        print(f"  [{level}] aucune donnee, figure non generee")
        return None

    fig, ax = plt.subplots(figsize=(17, 5.8))
    n_bands = len(BANDS)
    group_width = n_bands * BAND_STEP + GROUP_GAP
    offsets = [(di - 1) * WIDTH for di in range(len(DEFS))]  # -W, 0, +W

    n_sig = 0
    for g, state in enumerate(STATES_ORDERED):
        for bi, band in enumerate(BANDS):
            xc = g * group_width + bi * BAND_STEP
            for di, d in enumerate(DEFS):
                c = cells.get((state, band, di))
                if c is None:
                    continue
                x = xc + offsets[di]
                ax.bar(x, c["acc"], WIDTH,
                       color=BAND_COLORS[band], hatch=d["hatch"],
                       edgecolor="white", linewidth=0.6,
                       yerr=c["std"], capsize=1.5, error_kw=dict(lw=0.8))
                if c["sig"]:
                    n_sig += 1
                    ax.text(x, c["acc"] + c["std"] + 0.3, "*", ha="center",
                            va="bottom", fontsize=13, fontweight="bold")
                # Seuil par barre (raw, arthur) : chaque feature a sa loi nulle.
                if level in ("raw", "arthur") and not np.isnan(c["thr"]):
                    ax.plot([x - WIDTH / 2, x + WIDTH / 2], [c["thr"]] * 2,
                            "k--", lw=1.0, zorder=3)

        # En pooled, un trait par famille couvrant les 5 bandes du stade : le
        # seuil est commun aux 95 tests de la famille, pas a une bande.
        if level == "pooled":
            for di in range(len(DEFS)):
                t = pool_thr.get((state, di), np.nan)
                if np.isnan(t):
                    continue
                x0 = g * group_width + offsets[di] - WIDTH / 2
                x1 = g * group_width + (n_bands - 1) * BAND_STEP + offsets[di] + WIDTH / 2
                ax.plot([x0, x1], [t, t], "k--", lw=1.0, alpha=0.75, zorder=3)

    vals = [c["acc"] for c in cells.values()]
    lo = args.ymin if args.ymin is not None else min(48, min(vals) - 3)
    hi = args.ymax if args.ymax is not None else max(vals) + 5
    ax.set_ylim(lo, hi)

    # Abscisse : une etiquette de bande par triplet, un nom de stade dessous.
    ticks = [g * group_width + bi * BAND_STEP
             for g in range(len(STATES_ORDERED)) for bi in range(n_bands)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(BANDS * len(STATES_ORDERED), fontsize=8)
    for g, state in enumerate(STATES_ORDERED):
        ax.text(g * group_width + (n_bands - 1) * BAND_STEP / 2, lo - (hi - lo) * 0.085,
                state, ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_ylabel("Decoding accuracy (%)")
    ax.set_title(f"Definitions de la puissance spectrale, {LEVELS[level]}, p < {args.alpha}")
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Legende : les hachures portent la definition, la couleur porte la bande.
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="0.75", hatch=d["hatch"], edgecolor="white",
                       label=d["label"]) for d in DEFS],
        frameon=False, fontsize=9, ncol=3, loc="upper right")

    note = ("- - -  seuil pooled par famille   |   *  p < %.2g" % args.alpha
            if level == "pooled"
            else "- - -  seuil par feature   |   *  p < %.2g" % args.alpha)
    ax.text(0.995, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.3")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"barplot_psd_definitions_{level}_p{args.alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{level}] {n_sig} barres significatives -> {out}")
    return out


def print_table(args):
    """Tableau texte : accuracy et electrode par definition, pour lecture rapide."""
    cells, _ = collect(args, "raw")
    print("\n=== accuracy meilleure electrode (%) ===")
    print(f"{'state':6s} {'band':7s} {'brute':>16s} {'ratio':>16s} {'soustraction':>16s}")
    print("-" * 64)
    for state in STATES_ORDERED:
        for band in BANDS:
            row = f"{state:6s} {band:7s}"
            for di in range(len(DEFS)):
                c = cells.get((state, band, di))
                row += f" {c['acc']:10.2f} {c['elec']:>5s}" if c else f" {'--':>16s}"
            print(row)


def main() -> None:
    args = parse_args()
    print(f"=== barplots comparaison definitions PSD (p < {args.alpha}) ===")
    for level in ("raw", "arthur", "pooled"):
        make_figure(args, level)
    print_table(args)


if __name__ == "__main__":
    main()
