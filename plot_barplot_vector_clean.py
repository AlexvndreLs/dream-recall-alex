"""Trois barplots des features vectorielles modernisees (au-dela d'Arthur), par stade.

Features tracees : psd_osc x 5 bandes (FOOOF oscillatoire), aperiodic (1/f FOOOF),
higuchi_fd, perm_entropy, spec_entropy (complexite). Une barre par feature, hauteur
= accuracy de la MEILLEURE electrode (comme la Fig. 4 d'Arthur retient le pic par
electrode), error bar = acc_std inter-bootstrap, ligne de chance a 50 %.

Schema de permutation : SUBJECT (RFX) uniquement.

Trois figures, une par niveau de correction (l'etoile marque p < alpha) :

  raw    : p non corrigee de la meilleure electrode (d["pvals"][best]).
           Aucune correction. Le plus permissif. 9 features tracees.

  arthur : max-stat sur les 19 electrodes de la feature SEULE
           (compute_maxstat_correction.py --mode arthur -> *_maxstat_arthur.npz).
           9 features tracees.

  pooled : psd_osc corrige sur le POOL des 5 bandes (95 tests, un seul max-stat
           commun -> psd_osc_{state}_maxstat.npz). SEULES les 5 psd_osc sont
           tracees : les 4 complexites isolees n'ont pas de vrai pooling
           (pooled == arthur), on ne les affiche donc pas dans cette figure.

Traits pointilles (seuil, style Riemannian k--) :
  raw, arthur : un trait par barre, seuil propre a chaque feature (chaque feature
                a sa propre loi nulle).
  pooled      : un trait commun aux 5 barres psd_osc (seuil pooled partage).

Prerequis (deja generes via compute_maxstat_correction.py) :
    CORR=/scratch/alouis/dream_features_noica_1000hz_overlap_corrected
    # arthur (9 features) :
    python compute_maxstat_correction.py --save-path $SRC --output-path $CORR \
        --family-name unused --mode arthur \
        --keys psd_osc_delta psd_osc_theta psd_osc_alpha psd_osc_sigma psd_osc_beta \
               aperiodic higuchi_fd perm_entropy spec_entropy
    # pooled psd_osc (5 bandes) :
    python compute_maxstat_correction.py --save-path $SRC --output-path $CORR \
        --family-name psd_osc \
        --keys psd_osc_delta psd_osc_theta psd_osc_alpha psd_osc_sigma psd_osc_beta

Usage :
    python plot_barplot_vector_clean.py \
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

# Features vectorielles modernisees, dans l'ordre d'affichage.
PSD_OSC_KEYS = [f"psd_osc_{b}" for b in ("delta", "theta", "alpha", "sigma", "beta")]
COMPLEXITY_KEYS = ["aperiodic", "higuchi_fd", "perm_entropy", "spec_entropy"]
VECTOR_KEYS = PSD_OSC_KEYS + COMPLEXITY_KEYS


def keys_for_level(level: str):
    """Features tracees selon le niveau.

    raw / arthur : les 9 features (psd_osc x5 + 4 complexites).
    pooled       : uniquement les 5 psd_osc. Les complexites isolees n'ont pas de
                   vrai pooling (pooled == arthur), donc on ne les affiche pas ici :
                   la figure pooled ne montre que les features reellement corrigees
                   sur une famille.
    """
    return list(PSD_OSC_KEYS) if level == "pooled" else list(VECTOR_KEYS)


# psd_osc_* reutilisent les couleurs de bande de plot_common (via key_color).
# Les 4 complexites ne finissent par aucune bande : key_color renverrait le meme
# gris fallback pour les 4. On leur donne des couleurs distinctes ici.
COMPLEXITY_COLORS = {
    "aperiodic": "#17becf",
    "higuchi_fd": "#bcbd22",
    "perm_entropy": "#e377c2",
    "spec_entropy": "#7f7f7f",
}

WIDTH = 0.90
Y_LABEL = "Decoding accuracy (%)"

# Titres par niveau de correction.
LEVELS = {
    "raw":    "non corrige (p brute, meilleure electrode)",
    "arthur": "max-stat electrodes (Arthur, feature seule)",
    "pooled": "max-stat pooled, psd_osc sur 5 bandes",
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


def feature_color(key: str) -> str:
    if key in COMPLEXITY_COLORS:
        return COMPLEXITY_COLORS[key]
    return key_color(key)


def load_arthur_pval(corrected_path: Path, key: str, state: str, best_idx: int):
    """p corrigee max-stat (mode arthur) de la meilleure electrode, ou None."""
    f = corrected_path / f"{key}_{state}_maxstat_arthur.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return float(d["pvals_corrected"][best_idx])


def load_pooled_pval(corrected_path: Path, key: str, state: str):
    """p corrigee pooled de la meilleure electrode d'une feature psd_osc.

    Lit psd_osc_{state}_maxstat.npz (pool des 5 bandes). Les test_labels sont de
    la forme 'psd_osc_beta/O1' ; on prend, parmi les labels de cette key, la
    p-value minimale (= meilleure electrode retenue dans le pool). None si absent.
    """
    f = corrected_path / f"psd_osc_{state}_maxstat.npz"
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
    """Seuil d'accuracy (%) au quantile (1-alpha) de la nulle pooled psd_osc."""
    f = corrected_path / f"psd_osc_{state}_maxstat.npz"
    if not f.exists():
        return np.nan
    null_max = np.load(f, allow_pickle=True)["null_max"]
    return maxstat_threshold(null_max, alpha) * 100


def bar_threshold(save_path, corrected_path, key, state, best, level, alpha):
    """Seuil d'accuracy (%) que la barre (best electrode) doit depasser pour etre
    significative a ce niveau. Un seuil PROPRE A CHAQUE BARRE :

      raw    : quantile (1-alpha) de la nulle de la best electrode seule
               (perm_accs[:, best]), seuil non corrige de cette electrode.
      arthur : quantile (1-alpha) de null_max (max sur 19 electrodes) de la
               feature seule, seuil max-stat de cette feature.

    Retourne NaN si la donnee manque (pas de trait plutot que crash).
    """
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
    """Pour chaque (stade, feature) : accuracy best-elec, std, significativite,
    et seuil par barre (raw/arthur).

    Retourne accs[state][feat], stds[...], sigs[...], bar_thr[...] (seuil par
    barre, NaN en pooled), et thr_psdosc[state] (seuil groupe pooled, NaN sinon).
    """
    accs, stds, sigs, bar_thr, thr_psdosc = [], [], [], [], []
    for state in STATES_ORDERED:
        a_row, s_row, sig_row, t_row = [], [], [], []
        for key in keys_for_level(level):
            d = load_result(save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                a_row.append(np.nan); s_row.append(np.nan)
                sig_row.append(False); t_row.append(np.nan)
                continue

            acc = np.array(d["acc_mean"], dtype=float)  # (19,)
            std = np.array(d["acc_std"], dtype=float)   # (19,) : un std par electrode
            best = int(np.argmax(acc))
            a_row.append(acc[best] * 100)
            s_row.append(float(std[best]) * 100)

            if level == "raw":
                pvals = np.array(d["pvals"], dtype=float)
                p = float(pvals[best])
            elif level == "arthur":
                p = load_arthur_pval(corrected_path, key, state, best)
            else:  # pooled
                if key in PSD_OSC_KEYS:
                    p = load_pooled_pval(corrected_path, key, state)
                else:
                    # complexite isolee : pooled == arthur (feature seule)
                    p = load_arthur_pval(corrected_path, key, state, best)

            sig_row.append(p is not None and p < alpha)
            t_row.append(bar_threshold(save_path, corrected_path, key, state,
                                       best, level, alpha))

        accs.append(a_row); stds.append(s_row); sigs.append(sig_row)
        bar_thr.append(t_row)
        thr_psdosc.append(
            pooled_threshold(corrected_path, state, alpha) if level == "pooled" else np.nan
        )
    return accs, stds, sigs, bar_thr, thr_psdosc


def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax):
    accs, stds, sigs, bar_thr, thr_psdosc = collect(save_path, corrected_path, level, alpha)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    keys = keys_for_level(level)
    n_keys = len(keys)
    group_width = n_keys + 1  # une barre vide entre stades

    handles, seen = [], set()
    n_sig = 0
    for g, state in enumerate(STATES_ORDERED):
        for i, key in enumerate(keys):
            val = accs[g][i]
            if np.isnan(val):
                continue
            x = g * group_width + i
            b = ax.bar(x, val, WIDTH, color=feature_color(key),
                       yerr=stds[g][i], capsize=2, error_kw=dict(lw=1))
            if key not in seen:
                handles.append(b); seen.add(key)
            if sigs[g][i]:
                n_sig += 1
                ax.text(x, val + stds[g][i] + 0.4, "*", ha="center", va="bottom",
                        fontsize=15, fontweight="bold")

            # Trait par barre (raw, arthur) : seuil propre a cette feature, trace
            # sur la largeur de la barre. Style Riemannian (k--). Le pooled a son
            # propre trait groupe, gere plus bas.
            if level in ("raw", "arthur") and not np.isnan(bar_thr[g][i]):
                t = bar_thr[g][i]
                ax.plot([x - WIDTH / 2, x + WIDTH / 2], [t, t],
                        "k--", lw=1.2, zorder=3)

        # Trait pointille groupe : seulement pour pooled, sous-groupe psd_osc.
        if level == "pooled" and not np.isnan(thr_psdosc[g]):
            x0 = g * group_width - WIDTH / 2
            x1 = g * group_width + (len(PSD_OSC_KEYS) - 1) + WIDTH / 2
            ax.plot([x0, x1], [thr_psdosc[g], thr_psdosc[g]], "k--", lw=1.2, zorder=3)

    finite = [v for row in accs for v in row if not np.isnan(v)]
    lo = ymin if ymin is not None else min(48, min(finite) - 3)
    hi = ymax if ymax is not None else max(finite) + 5
    ax.set_ylim(lo, hi)

    ax.set_ylabel(Y_LABEL)
    ax.set_title(
        f"Features vectorielles modernisees (subject/RFX), {LEVELS[level]}, p < {alpha}"
    )
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATES_ORDERED))])
    ax.set_xticklabels(STATES_ORDERED)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    if handles:
        ax.legend(handles, [band_label(k) for k in keys],
                  frameon=False, fontsize=8, ncol=3,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))

    note = "*  p < %.2g" % alpha
    if level == "pooled":
        note = "- - -  seuil pooled psd_osc   |   " + note
    else:  # raw, arthur : un seuil par barre
        note = "- - -  seuil par feature   |   " + note
    ax.text(0.995, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.3")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"barplot_vector_{level}_p{alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"  [{level}] {n_sig} barres significatives -> {out}")
    return out


def main() -> None:
    args = parse_args()
    print(f"=== barplots vectoriels (subject/RFX, p < {args.alpha}) ===")
    for level in ("raw", "arthur", "pooled"):
        make_figure(args.save_path, args.corrected_path, args.out_dir,
                    level, args.alpha, args.ymin, args.ymax)


if __name__ == "__main__":
    main()
