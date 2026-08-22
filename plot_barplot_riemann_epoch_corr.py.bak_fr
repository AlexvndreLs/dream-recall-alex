"""Barplot des accuracies riemanniennes, schéma de permutation EPOCH, avec
correction pour comparaisons multiples paramétrable.

Même contenu que plot_barplot_riemann_arthur.py --perm-scheme epoch (accuracy =
acc_mean sur les 1000 bootstraps, barres d'erreur = acc_std), avec en plus le
choix de la portée de la correction max-stat :

    none   : un seuil par combo, non corrigé (comportement d'origine)
    state  : max-stat sur les 6 features du stade, un seuil par stade
             (ce que la Fig. 2 de la thèse annonce : "corrected for multiple
             comparisons using maximum statistics")
    global : max-stat sur les 24 combos, un seuil unique pour toute la figure

Le max-stat empile les lois nulles de la famille de tests, prend le maximum sur
ces tests à chaque permutation, puis seuille la distribution des maxima ainsi
obtenue. Plus la famille est large, plus le seuil monte.

Lit les fichiers *_epochperm.npz (schéma epoch d'Arthur, avec re-tirage du
sous-échantillon à chaque permutation). Pour la réplication stricte avec
sous-échantillon figé, voir plot_barplot_riemann_ffx_fixed.py.

Usage :
    python plot_barplot_riemann_epoch_corr.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --correction global \
        --out-dir plot_overlap/ \
        --alpha 0.001
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans affichage, obligatoire sur noeud de calcul
import matplotlib.pyplot as plt
import numpy as np

from config_v3 import FREQ_DICT

STATE_LIST = ["S2", "SWS", "NREM", "REM"]  # ordre figure : NREM avant REM

# ─── paramètres figure (repris de visu_barplot_cosp.py d'Arthur) ─────────────

MINMAX = [40, 80]
Y_LABEL = "Decoding accuracies (%)"
GRAPH_TITLE = "Riemannian classifications, perm. epoch"
WIDTH = 0.90
RESOLUTION = 300

# Gris pour la covariance, puis une couleur par bande de fréquence.
COLORS = ["#C2C2C2", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, contenant results/.")
    p.add_argument("--out-dir", type=Path, default=Path("figures"),
                   help="Dossier de sortie de la figure.")
    p.add_argument("--alpha", type=float, default=0.001,
                   help="Seuil de significativité tracé en pointillés.")
    p.add_argument("--correction", choices=["none", "state", "global"], default="none",
                   help="none   : un seuil par combo, non corrigé. "
                        "state  : max-stat sur les 6 features du stade. "
                        "global : max-stat sur les 24 combos, seuil unique.")
    return p.parse_args()


def result_path(save_path: Path, key: str, state: str) -> Path:
    """Chemin du .npz schéma epoch (suffixe _epochperm)."""
    return save_path / "results" / f"{key}_{state}_epochperm.npz"


def perm_threshold(perm_accs: np.ndarray, alpha: float) -> float:
    """Seuil d'accuracy au quantile (1 - alpha) de la loi nulle.

    Identique au calcul d'Arthur (visu_barplot_cosp.py) :
        ind = int(alpha * len(pscores)); threshold = sorted(pscores)[-ind]
    """
    ind = max(1, int(alpha * len(perm_accs)))
    return float(np.sort(perm_accs)[-ind])


def load_one(path: Path) -> tuple[float, float, np.ndarray | None] | None:
    """Charge (accuracy moyenne, écart-type, loi nulle brute).

    acc_scores contient une accuracy par bootstrap, chacune déjà moyennée sur
    les splits de la CV. acc_std est donc la dispersion inter-bootstrap.

    La nulle est retournée brute plutôt que réduite à un seuil, pour permettre
    la correction max-stat qui a besoin des distributions complètes avant de
    calculer quoi que ce soit.
    """
    if not path.exists():
        print(f"  absent : {path.name}")
        return None

    d = np.load(path, allow_pickle=True)
    acc = float(d["acc_mean"]) * 100
    std = float(d["acc_std"]) * 100

    if "perm_accs" not in d:
        print(f"  pas de perm_accs : {path.name}")
        return acc, std, None

    return acc, std, np.asarray(d["perm_accs"])


def main() -> None:
    args = parse_args()

    # Covariance en premier, puis les cospectres, comme chez Arthur.
    keys = ["cov"] + [f"cosp_{b}" for b in FREQ_DICT]
    legend_labels = ["Covariance"] + [f"{b} cospec" for b in FREQ_DICT]

    MODE_LABEL = {
        "none": "par combo (non corrigé)",
        "state": "max-stat par stade, 6 features",
        "global": "max-stat global, 24 combos",
    }
    print("=== barplot riemannien, schéma EPOCH ===")
    print(f"seuil : {MODE_LABEL[args.correction]}\n")

    # ── 1. chargement, tous stades ───────────────────────────────────────────
    accs, stds, all_nulls = [], [], []
    for state in STATE_LIST:
        a_row, s_row, n_row = [], [], []
        for key in keys:
            res = load_one(result_path(args.save_path, key, state))
            if res is None:
                a_row.append(np.nan)
                s_row.append(np.nan)
                n_row.append(None)
            else:
                acc, std, null = res
                a_row.append(acc)
                s_row.append(std)
                n_row.append(null)
        accs.append(a_row)
        stds.append(s_row)
        all_nulls.append(n_row)

    # ── 2. seuils selon la portée de correction ──────────────────────────────
    thresholds = []

    if args.correction == "global":
        pooled = np.array([n for row in all_nulls for n in row if n is not None])
        if len(pooled) == 0:
            raise RuntimeError("aucune loi nulle chargée, rien à corriger.")
        thr = perm_threshold(pooled.max(axis=0), args.alpha) * 100
        print(f"  seuil global = {thr:.2f}%  ({len(pooled)} combos poolés)")
        thresholds = [[thr if n is not None else np.nan for n in row]
                      for row in all_nulls]
    else:
        for state, n_row in zip(STATE_LIST, all_nulls):
            if args.correction == "state":
                stacked = np.array([n for n in n_row if n is not None])
                if len(stacked) == 0:
                    t_row = [np.nan] * len(keys)
                else:
                    thr = perm_threshold(stacked.max(axis=0), args.alpha) * 100
                    t_row = [thr if n is not None else np.nan for n in n_row]
                    print(f"  {state:5s} : seuil max-stat = {thr:.2f}%  "
                          f"({len(stacked)} features poolées)")
            else:
                t_row = [perm_threshold(n, args.alpha) * 100 if n is not None
                         else np.nan for n in n_row]
                valid = [t for t in t_row if not np.isnan(t)]
                if valid:
                    print(f"  {state:5s} : seuils {min(valid):.2f}–{max(valid):.2f}%"
                          f"  (étendue {max(valid) - min(valid):.2f} pts)")
            thresholds.append(t_row)

    n_sig = sum(1 for g in range(len(STATE_LIST)) for i in range(len(keys))
                if not np.isnan(accs[g][i]) and not np.isnan(thresholds[g][i])
                and accs[g][i] > thresholds[g][i])
    n_tot = sum(1 for g in range(len(STATE_LIST)) for i in range(len(keys))
                if not np.isnan(accs[g][i]))
    print(f"\n  significatifs : {n_sig}/{n_tot} combos")

    # ── 3. figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    n_keys = len(keys)
    group_width = n_keys + 1
    bars = []
    for g, state in enumerate(STATE_LIST):
        for i in range(n_keys):
            x = g * group_width + i
            val = accs[g][i]
            if np.isnan(val):
                continue
            b = ax.bar(x, val, WIDTH, color=COLORS[i], yerr=stds[g][i],
                       capsize=2, error_kw=dict(lw=1))
            if g == 0:
                bars.append(b)

            # En mode global le seuil est unique : une ligne continue traversant
            # la figure est plus lisible que 24 traits identiques.
            t = thresholds[g][i]
            if not np.isnan(t):
                if args.correction != "global":
                    ax.plot([x - WIDTH / 2, x + WIDTH / 2], [t, t], "k--", lw=1)
                if val > t:
                    ax.text(x, val + stds[g][i] + 0.5, "*", ha="center",
                            va="bottom", fontsize=14, fontweight="bold")

    if args.correction == "global":
        gthr = next((t for row in thresholds for t in row if not np.isnan(t)), None)
        if gthr is not None:
            ax.axhline(gthr, color="k", ls="--", lw=1)

    CORR_TITLE = {"none": "", "state": ", max-stat par stade",
                  "global": ", max-stat global (24 combos)"}
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(MINMAX)
    ax.set_title(f"{GRAPH_TITLE}{CORR_TITLE[args.correction]}, p < {args.alpha}")
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATE_LIST))])
    ax.set_xticklabels(STATE_LIST)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)  # niveau de chance

    if bars:
        ax.legend(bars, legend_labels, frameon=False, fontsize=9,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))

    CORR_SUFFIX = {"none": "", "state": "_maxstat_state", "global": "_maxstat_global"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = (args.out_dir /
           f"barplot_riemann_epoch{CORR_SUFFIX[args.correction]}_p{args.alpha}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()