"""Trois barplots des PSD bruts (features vectorielles d'origine, facon Arthur), par stade.

Features tracees : psd x 5 bandes (delta, theta, alpha, sigma, beta) : la puissance
spectrale par bande, exactement les features du chapitre 1 d'Arthur (avant FOOOF /
complexite). Une barre par bande, hauteur = accuracy de la MEILLEURE electrode
(comme la Fig. 4 d'Arthur retient le pic par electrode), error bar = acc_std
inter-bootstrap, ligne de chance a 50 %.

Schema de permutation : SUBJECT (RFX) uniquement. C'est la difference centrale avec
Arthur (schema epoch) : sur cette branche, seul psd_sigma/S2 resiste a la correction
par feature, illustrant l'effet du schema de permutation sur les conclusions.

Trois figures, une par niveau de correction (l'etoile marque p < alpha) :

  raw    : p non corrigee de la meilleure electrode (d["pvals"][best]).
  arthur : max-stat sur les 19 electrodes de la feature SEULE (*_maxstat_arthur.npz).
  pooled : max-stat sur le POOL des 5 bandes psd (95 tests, psd_{state}_maxstat.npz).

Traits pointilles (style Riemannian k--) :
  raw, arthur : un trait par barre (seuil propre a chaque feature).
  pooled      : un trait commun aux 5 barres (seuil pooled partage).

Prerequis (compute_maxstat_correction.py) :
    CORR=/scratch/alouis/dream_features_noica_1000hz_overlap_corrected
    # arthur (5 bandes) :
    python compute_maxstat_correction.py --save-path $SRC --output-path $CORR \
        --family-name unused --mode arthur \
        --keys psd_delta psd_theta psd_alpha psd_sigma psd_beta
    # pooled psd (5 bandes) :
    python compute_maxstat_correction.py --save-path $SRC --output-path $CORR \
        --family-name psd \
        --keys psd_delta psd_theta psd_alpha psd_sigma psd_beta

Usage :
    python plot_barplot_psd_arthur_clean.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
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
    RESOLUTION,
    STATES_ORDERED,
    band_label,
    key_color,
    load_result,
    maxstat_threshold,
)

PSD_KEYS = [f"psd_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
POOL_FAMILY = "psd"  # -> psd_{state}_maxstat.npz

WIDTH = 0.90
Y_LABEL = "Decoding accuracy (%)"

LEVELS = {
    "raw":    "non corrige (p brute, meilleure electrode)",
    "arthur": "max-stat electrodes (Arthur, feature seule)",
    "pooled": "max-stat pooled, psd sur 5 bandes",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, contenant results/.")
    p.add_argument("--corrected-path", type=Path, required=True,
                   help="Dossier des .npz maxstat (compute_maxstat_correction.py).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Dossier de sortie des figures.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    return p.parse_args()


def load_arthur_pval(corrected_path: Path, key: str, state: str, best_idx: int):
    """p corrigee max-stat (mode arthur) de la meilleure electrode, ou None."""
    f = corrected_path / f"{key}_{state}_maxstat_arthur.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return float(d["pvals_corrected"][best_idx])


def load_pooled_pval(corrected_path: Path, key: str, state: str):
    """p corrigee pooled de la meilleure electrode d'une bande psd.

    Lit psd_{state}_maxstat.npz (pool des 5 bandes). test_labels de la forme
    'psd_sigma/Fp2' ; on prend la p minimale parmi les labels de cette key
    (= meilleure electrode retenue dans le pool). None si absent.
    """
    f = corrected_path / f"{POOL_FAMILY}_{state}_maxstat.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    labels = [str(x) for x in d["test_labels"]]
    pvals = np.array(d["pvals_corrected"])
    mask = [lab.startswith(f"{key}/") for lab in labels]
    if not any(mask):
        return None
    return float(pvals[np.array(mask)].min())


def pooled_threshold(corrected_path: Path, state: str, alpha: float):
    """Seuil d'accuracy (%) au quantile (1-alpha) de la nulle pooled psd."""
    f = corrected_path / f"{POOL_FAMILY}_{state}_maxstat.npz"
    if not f.exists():
        return np.nan
    null_max = np.load(f, allow_pickle=True)["null_max"]
    return maxstat_threshold(null_max, alpha) * 100


def bar_threshold(save_path, corrected_path, key, state, best, level, alpha):
    """Seuil d'accuracy (%) propre a chaque barre (raw / arthur). NaN si absent."""
    if level == "raw":
        d = load_result(save_path, key, state)
        if d is None or "perm_accs" not in d:
            return np.nan
        null_best = np.array(d["perm_accs"])[:, best]
        return maxstat_threshold(null_best, alpha) * 100
    if level == "arthur":
        f = corrected_path / f"{key}_{state}_maxstat_arthur.npz"
        if not f.exists():
            return np.nan
        null_max = np.load(f, allow_pickle=True)["null_max"]
        return maxstat_threshold(null_max, alpha) * 100
    return np.nan


def collect(save_path: Path, corrected_path: Path, level: str, alpha: float):
    accs, stds, sigs, bar_thr, thr_pool = [], [], [], [], []
    for state in STATES_ORDERED:
        a_row, s_row, sig_row, t_row = [], [], [], []
        for key in PSD_KEYS:
            d = load_result(save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                a_row.append(np.nan); s_row.append(np.nan)
                sig_row.append(False); t_row.append(np.nan)
                continue

            acc = np.array(d["acc_mean"], dtype=float)  # (19,)
            std = np.array(d["acc_std"], dtype=float)   # (19,)
            best = int(np.argmax(acc))
            a_row.append(acc[best] * 100)
            s_row.append(float(std[best]) * 100)

            if level == "raw":
                p = float(np.array(d["pvals"], dtype=float)[best])
            elif level == "arthur":
                p = load_arthur_pval(corrected_path, key, state, best)
            else:  # pooled
                p = load_pooled_pval(corrected_path, key, state)

            sig_row.append(p is not None and p < alpha)
            t_row.append(bar_threshold(save_path, corrected_path, key, state,
                                       best, level, alpha))

        accs.append(a_row); stds.append(s_row); sigs.append(sig_row)
        bar_thr.append(t_row)
        thr_pool.append(
            pooled_threshold(corrected_path, state, alpha) if level == "pooled" else np.nan
        )
    return accs, stds, sigs, bar_thr, thr_pool


def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax):
    accs, stds, sigs, bar_thr, thr_pool = collect(save_path, corrected_path, level, alpha)

    fig, ax = plt.subplots(figsize=(11, 5))
    n_keys = len(PSD_KEYS)
    group_width = n_keys + 1

    handles = []
    n_sig = 0
    for g, state in enumerate(STATES_ORDERED):
        for i, key in enumerate(PSD_KEYS):
            val = accs[g][i]
            if np.isnan(val):
                continue
            x = g * group_width + i
            b = ax.bar(x, val, WIDTH, color=key_color(key),
                       yerr=stds[g][i], capsize=2, error_kw=dict(lw=1))
            if g == 0:
                handles.append(b)
            if sigs[g][i]:
                n_sig += 1
                ax.text(x, val + stds[g][i] + 0.4, "*", ha="center", va="bottom",
                        fontsize=15, fontweight="bold")

            if level in ("raw", "arthur") and not np.isnan(bar_thr[g][i]):
                t = bar_thr[g][i]
                ax.plot([x - WIDTH / 2, x + WIDTH / 2], [t, t], "k--", lw=1.2, zorder=3)

        if level == "pooled" and not np.isnan(thr_pool[g]):
            x0 = g * group_width - WIDTH / 2
            x1 = g * group_width + (n_keys - 1) + WIDTH / 2
            ax.plot([x0, x1], [thr_pool[g], thr_pool[g]], "k--", lw=1.2, zorder=3)

    finite = [v for row in accs for v in row if not np.isnan(v)]
    lo = ymin if ymin is not None else min(48, min(finite) - 3)
    hi = ymax if ymax is not None else max(finite) + 5
    ax.set_ylim(lo, hi)

    ax.set_ylabel(Y_LABEL)
    ax.set_title(
        f"PSD bruts facon Arthur (subject/RFX), {LEVELS[level]}, p < {alpha}"
    )
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATES_ORDERED))])
    ax.set_xticklabels(STATES_ORDERED)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    if handles:
        ax.legend(handles, [band_label(k) for k in PSD_KEYS],
                  frameon=False, fontsize=9, ncol=2,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))

    if level == "pooled":
        note = "- - -  seuil pooled psd   |   *  p < %.2g corrige" % alpha
    else:
        note = "- - -  seuil par feature   |   *  p < %.2g" % alpha
    ax.text(0.995, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.3")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"barplot_psd_arthur_{level}_p{alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"  [{level}] {n_sig} barres significatives sur "
          f"{n_keys * len(STATES_ORDERED)} -> {out}")
    return out


def main() -> None:
    args = parse_args()
    print(f"=== barplots psd bruts (subject/RFX, p < {args.alpha}) ===")
    for level in ("raw", "arthur", "pooled"):
        make_figure(args.save_path, args.corrected_path, args.out_dir,
                    level, args.alpha, args.ymin, args.ymax)


if __name__ == "__main__":
    main()
