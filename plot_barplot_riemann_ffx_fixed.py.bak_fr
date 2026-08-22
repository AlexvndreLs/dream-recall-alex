"""Barplot des accuracies riemanniennes, schéma de permutation EPOCH FIXÉ.

Réplication stricte du protocole d'Arthur Dehgan (classif_cosp.py:95-107 +
utils.py:99-111) : le sous-échantillon est tiré UNE SEULE FOIS, puis seuls les
labels sont permutés sur ce X figé. Lit les fichiers *_epochperm_fixed.npz
produits par recompute_perms_epoch_arthur_fixed.py.

DIFFÉRENCE CLÉ avec plot_barplot_riemann_arthur.py :
    Ce script trace acc_obs_fixed (CV sur le X figé), PAS acc_mean (moyenne des
    1000 bootstraps du schéma subject). C'est la seule accuracy cohérente avec
    la loi nulle du fixed : observé et nulle proviennent du même échantillon.
    Tracer acc_mean contre le seuil du fixed réintroduirait exactement
    l'incohérence observé/nulle que le schéma fixé corrige.

    Conséquence visuelle attendue : la nulle étant construite sur un X unique,
    les seuils de permutation s'alignent horizontalement au sein d'un stade,
    au lieu d'être dispersés feature par feature comme dans l'ancien schéma
    epoch. Ce désalignement est précisément ce qui avait révélé le problème.

    Pas de barre d'erreur : un seul échantillon figé, aucune dispersion
    inter-bootstrap à afficher.

Le trait pointillé est le seuil de significativité issu de la loi nulle :
    seuil = sorted(perm_accs)[-int(alpha * n_perm)]
identique au calcul d'Arthur (visu_barplot_cosp.py). À alpha=0.001 sur 1000
permutations, ind=1 : le seuil est le MAXIMUM de la nulle, donc une barre qui
le dépasse signifie qu'aucune permutation n'a atteint l'accuracy observée.

Les combos absents de results/ sont signalés et laissés vides plutôt que de
faire échouer la figure.

Usage :
    python plot_barplot_riemann_ffx_fixed.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
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
GRAPH_TITLE = "Riemannian classifications, FFX (X figé, réplication Arthur)"
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
                        "state  : max-stat sur les 6 features du stade, un seuil "
                        "par stade (Arthur, Fig. 2). "
                        "global : max-stat sur les 24 combos, un seuil unique "
                        "pour toute la figure.")
    return p.parse_args()


def result_path(save_path: Path, key: str, state: str) -> Path:
    """Chemin du .npz schéma epoch fixé, écrit par
    recompute_perms_epoch_arthur_fixed.py."""
    return save_path / "results" / f"{key}_{state}_epochperm_fixed.npz"


def perm_threshold(perm_accs: np.ndarray, alpha: float) -> float:
    """Seuil d'accuracy au quantile (1 - alpha) de la loi nulle.

    Identique au calcul d'Arthur (visu_barplot_cosp.py) :
        ind = int(alpha * len(pscores)); threshold = sorted(pscores)[-ind]
    """
    ind = max(1, int(alpha * len(perm_accs)))
    return float(np.sort(perm_accs)[-ind])


def load_one(path: Path) -> tuple[float, np.ndarray | None] | None:
    """Charge (accuracy observée sur X figé en %, loi nulle brute).

    La nulle est retournée telle quelle plutôt que réduite à un seuil, pour
    permettre la correction max-stat qui a besoin des distributions complètes
    de toutes les features d'un stade avant de calculer quoi que ce soit.

    Retourne None si le fichier est absent : tous les combos ne sont pas
    forcément calculés. Plante explicitement si acc_obs_fixed manque, ce qui
    signalerait un fichier produit par l'ancien script et non par le fixed.
    """
    if not path.exists():
        print(f"  absent : {path.name}")
        return None

    d = np.load(path, allow_pickle=True)

    if "acc_obs_fixed" not in d:
        raise KeyError(
            f"{path.name} ne contient pas acc_obs_fixed — ce fichier ne vient "
            f"pas de recompute_perms_epoch_arthur_fixed.py. Utilise "
            f"plot_barplot_riemann_arthur.py pour les autres schémas."
        )

    acc = float(d["acc_obs_fixed"]) * 100

    if "perm_accs" not in d:
        print(f"  pas de perm_accs : {path.name}")
        return acc, None

    return acc, np.asarray(d["perm_accs"])


def main() -> None:
    args = parse_args()

    # Covariance en premier, puis les cospectres, comme chez Arthur.
    keys = ["cov"] + [f"cosp_{b}" for b in FREQ_DICT]
    legend_labels = ["Covariance"] + [f"{b} cospec" for b in FREQ_DICT]

    MODE_LABEL = {
        "none": "par combo (non corrigé)",
        "state": "max-stat par stade, 6 features (Arthur)",
        "global": "max-stat global, 24 combos",
    }
    print("=== barplot riemannien, schéma EPOCH FIXÉ (X figé) ===")
    print(f"accuracy tracée : acc_obs_fixed (CV sur le X figé, sans barre d'erreur)")
    print(f"seuil           : {MODE_LABEL[args.correction]}\n")

    # ── 1. chargement : accuracies et lois nulles brutes, tous stades ────────
    accs, all_nulls = [], []
    for state in STATE_LIST:
        a_row, n_row = [], []
        for key in keys:
            res = load_one(result_path(args.save_path, key, state))
            if res is None:
                a_row.append(np.nan)
                n_row.append(None)
            else:
                acc, null = res
                a_row.append(acc)
                n_row.append(null)
        accs.append(a_row)
        all_nulls.append(n_row)

    # ── 2. seuils, selon la portée de correction demandée ────────────────────
    # Le max-stat empile les lois nulles, prend le maximum sur les tests à
    # chaque permutation, puis seuille cette distribution des maxima. Plus la
    # famille de tests est large, plus le seuil monte.
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

    # Décompte des combos significatifs, utile pour la note méthodologique.
    n_sig = sum(1 for g in range(len(STATE_LIST)) for i in range(len(keys))
                if not np.isnan(accs[g][i]) and not np.isnan(thresholds[g][i])
                and accs[g][i] > thresholds[g][i])
    n_tot = sum(1 for g in range(len(STATE_LIST)) for i in range(len(keys))
                if not np.isnan(accs[g][i]))
    print(f"\n  significatifs : {n_sig}/{n_tot} combos")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Un groupe de barres par stade, séparés par un espace vide d'une barre.
    n_keys = len(keys)
    group_width = n_keys + 1
    bars = []
    for g, state in enumerate(STATE_LIST):
        for i in range(n_keys):
            x = g * group_width + i
            val = accs[g][i]
            if np.isnan(val):
                continue
            b = ax.bar(x, val, WIDTH, color=COLORS[i])
            if g == 0:
                bars.append(b)

            # Seuil de permutation. En mode global le seuil est unique, une
            # ligne continue traversant la figure est plus lisible que 24
            # traits identiques ; elle est tracée une seule fois plus bas.
            t = thresholds[g][i]
            if not np.isnan(t):
                if args.correction != "global":
                    ax.plot([x - WIDTH / 2, x + WIDTH / 2], [t, t], "k--", lw=1)
                if val > t:
                    ax.text(x, val + 0.5, "*", ha="center", va="bottom",
                            fontsize=14, fontweight="bold")

    if args.correction == "global":
        gthr = next((t for row in thresholds for t in row if not np.isnan(t)), None)
        if gthr is not None:
            ax.axhline(gthr, color="k", ls="--", lw=1,
                       label=f"p < {args.alpha} (max-stat, 24 combos)")

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(MINMAX)
    CORR_TITLE = {"none": "", "state": ", max-stat par stade",
                  "global": ", max-stat global (24 combos)"}
    ax.set_title(f"{GRAPH_TITLE}{CORR_TITLE[args.correction]}, p < {args.alpha}")
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATE_LIST))])
    ax.set_xticklabels(STATE_LIST)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)  # niveau de chance

    if bars:
        ax.legend(bars, legend_labels, frameon=False, fontsize=9,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    CORR_SUFFIX = {"none": "", "state": "_maxstat_state", "global": "_maxstat_global"}
    out = (args.out_dir /
           f"barplot_riemann_epoch_fixed{CORR_SUFFIX[args.correction]}_p{args.alpha}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()