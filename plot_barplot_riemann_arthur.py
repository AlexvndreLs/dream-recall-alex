"""Barplot des accuracies riemanniennes (covariance + cospectres), par stade.

Réplique la figure de classification riemannienne du chapitre 1 de la thèse
d'Arthur Dehgan (visu_barplot_cosp.py du repo arthurdehgan/sleep), adaptée au
format .npz produit par classify.py.

Lit les résultats du schéma de permutation EPOCH (fichiers *_epochperm.npz
produits par replicate_arthur_ffx.py), qui est le schéma d'Arthur
(utils.py:103). Le schéma subject de classify.py donne des seuils plus
conservateurs et n'est pas ce qui est répliqué ici.

Le trait pointillé au-dessus de chaque barre est le seuil de significativité
issu de la distribution nulle par permutation, au niveau alpha demandé :
    seuil = sorted(perm_accs)[-int(alpha * n_perm)]
identique au calcul d'Arthur. Une barre dépassant son trait est significative.

Les combos absents de results/ sont signalés et laissés vides plutôt que de
faire échouer la figure : l'inventaire des runs terminés n'est pas garanti
complet.

Usage :
    python plot_barplot_riemann.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir figures/ \
        --alpha 0.001
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans affichage, obligatoire sur noeud de calcul
import matplotlib.pyplot as plt
import numpy as np

from config_v3 import FREQ_DICT, STATE_LIST

# ─── paramètres figure (repris de visu_barplot_cosp.py d'Arthur) ─────────────

MINMAX = [40, 80]
Y_LABEL = "Decoding accuracies (%)"
GRAPH_TITLE = "Riemannian classifications"
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
    p.add_argument("--perm-scheme", choices=["epoch", "subject"], default="epoch",
                   help="epoch = réplication Arthur (*_epochperm.npz), "
                        "subject = schéma corrigé (*.npz).")
    return p.parse_args()


def result_path(save_path: Path, key: str, state: str, scheme: str) -> Path:
    """Chemin du .npz de résultats pour un couple (feature, stade).

    Le schéma epoch est écrit dans un fichier séparé par
    replicate_arthur_ffx.py, suffixe _epochperm, dans le même dossier.
    """
    suffix = "_epochperm" if scheme == "epoch" else ""
    return save_path / "results" / f"{key}_{state}{suffix}.npz"


def perm_threshold(perm_accs: np.ndarray, alpha: float) -> float:
    """Seuil d'accuracy correspondant au quantile (1 - alpha) de la loi nulle.

    Identique au calcul d'Arthur (visu_barplot_cosp.py) :
        ind = int(alpha * len(pscores)); threshold = sorted(pscores)[-ind]
    Avec alpha=0.001 et 1000 permutations, ind=1, le seuil est donc le maximum
    de la distribution nulle : aucune permutation ne l'a atteint.
    """
    ind = max(1, int(alpha * len(perm_accs)))
    return float(np.sort(perm_accs)[-ind])


def load_one(path: Path, alpha: float) -> tuple[float, float, float] | None:
    """Charge (accuracy moyenne, écart-type, seuil de permutation) en %.

    acc_scores contient une accuracy par bootstrap, chacune déjà moyennée sur
    les 324 splits de la CV par run_cv. acc_std est donc directement la
    dispersion inter-bootstrap, sans regroupement supplémentaire à faire.

    Retourne None si le fichier est absent : tous les combos ne sont pas
    forcément calculés.
    """
    if not path.exists():
        print(f"  absent : {path.name}")
        return None

    d = np.load(path, allow_pickle=True)
    acc = float(d["acc_mean"]) * 100
    std = float(d["acc_std"]) * 100

    if "perm_accs" not in d:
        print(f"  pas de perm_accs : {path.name}")
        return acc, std, np.nan

    return acc, std, perm_threshold(d["perm_accs"], alpha) * 100


def main() -> None:
    args = parse_args()

    # Covariance en premier, puis les cospectres, comme chez Arthur.
    keys = ["cov"] + [f"cosp_{b}" for b in FREQ_DICT]
    legend_labels = ["Covariance"] + [f"{b} cospec" for b in FREQ_DICT]

    print(f"=== barplot riemannien (schéma {args.perm_scheme}) ===")

    accs, stds, thresholds = [], [], []
    for state in STATE_LIST:
        a_row, s_row, t_row = [], [], []
        for key in keys:
            res = load_one(result_path(args.save_path, key, state, args.perm_scheme),
                           args.alpha)
            if res is None:
                a_row.append(np.nan)
                s_row.append(np.nan)
                t_row.append(np.nan)
            else:
                acc, std, thresh = res
                a_row.append(acc)
                s_row.append(std)
                t_row.append(thresh)
        accs.append(a_row)
        stds.append(s_row)
        thresholds.append(t_row)

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
            b = ax.bar(x, val, WIDTH, color=COLORS[i], yerr=stds[g][i],
                       capsize=2, error_kw=dict(lw=1))
            if g == 0:
                bars.append(b)

            # Seuil de permutation : trait pointillé sur la largeur de la barre,
            # calculé par combo comme chez Arthur (visu_barplot_cosp.py).
            t = thresholds[g][i]
            if not np.isnan(t):
                ax.plot([x - WIDTH / 2, x + WIDTH / 2], [t, t], "k--", lw=1)
                # Étoile au-dessus de la barre d'erreur si l'accuracy dépasse le
                # seuil (significatif au niveau alpha).
                if val > t:
                    ax.text(x, val + stds[g][i] + 0.5, "*", ha="center",
                            va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(MINMAX)
    ax.set_title(f"{GRAPH_TITLE}, perm. {args.perm_scheme}, p < {args.alpha}")
    ax.set_xticks([g * group_width + (n_keys - 1) / 2 for g in range(len(STATE_LIST))])
    ax.set_xticklabels(STATE_LIST)
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)  # niveau de chance

    if bars:
        ax.legend(bars, legend_labels, frameon=False, fontsize=9,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"barplot_riemann_{args.perm_scheme}_p{args.alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()