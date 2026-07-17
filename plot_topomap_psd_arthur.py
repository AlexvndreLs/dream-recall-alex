"""Topomaps des accuracies de décodage par électrode, pour les features PSD.

Réplique la figure de décodage univarié du chapitre 1 de la thèse d'Arthur
Dehgan (visu_topomap.py du repo arthurdehgan/sleep), adaptée au format .npz
produit par classify.py.

Chaque feature vectorielle est classifiée une électrode à la fois (LDA
univarié) : acc_mean est un vecteur de 19 accuracies, une par électrode. Cette
grille bandes × stades donne la distribution spatiale du pouvoir décodant,
c'est-à-dire l'équivalent d'une carte d'importance pour les features
vectorielles.

Les électrodes significatives sont marquées d'un point blanc. Le seuil est
issu de la distribution nulle par permutation, schéma EPOCH par défaut
(réplication d'Arthur, cf replicate_arthur_ffx.py).

Note sur la correction des comparaisons multiples : --correction maxstat
reproduit Arthur (visu_topomap.py, MAXSTAT_ELEC = True) : la loi nulle est le
maximum des scores de permutation sur les 19 électrodes, et chaque accuracy est
comparée à cette loi nulle du max, ce qui contrôle le FWER sur la carte.
--correction none donne la p brute par électrode, sans correction : ce n'est PAS
ce que fait la thèse, c'est fourni comme point de comparaison plus permissif.

Usage :
    python plot_topomap_psd.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir figures/ \
        --feature-family psd \
        --alpha 0.001
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from config_v3 import CH_NAMES, FREQ_DICT, N_EEG, STATE_LIST

RESOLUTION = 300


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, contenant results/.")
    p.add_argument("--out-dir", type=Path, default=Path("figures"),
                   help="Dossier de sortie de la figure.")
    p.add_argument("--feature-family", default="psd",
                   choices=["psd", "psd_osc"],
                   help="psd = puissance brute (Arthur), psd_osc = ratio "
                        "puissance/aperiodic (extension FOOOF).")
    p.add_argument("--alpha", type=float, default=0.001,
                   help="Seuil de significativité pour marquer les électrodes.")
    p.add_argument("--perm-scheme", choices=["epoch", "subject"], default="epoch",
                   help="epoch = réplication Arthur (*_epochperm.npz), "
                        "subject = schéma corrigé (*.npz).")
    p.add_argument("--correction", choices=["none", "maxstat"], default="maxstat",
                   help="maxstat = correction FWER sur les 19 électrodes "
                        "(Arthur, MAXSTAT_ELEC=True), none = p brute par "
                        "électrode, non corrigée.")
    p.add_argument("--vmin", type=float, default=None,
                   help="Borne basse de l'échelle de couleur (%%). "
                        "Par défaut : min des données.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Borne haute de l'échelle de couleur (%%).")
    return p.parse_args()


def result_path(save_path: Path, key: str, state: str, scheme: str) -> Path:
    """Chemin du .npz de résultats pour un couple (feature, stade)."""
    suffix = "_epochperm" if scheme == "epoch" else ""
    return save_path / "results" / f"{key}_{state}{suffix}.npz"


def make_info() -> mne.Info:
    """Info MNE portant les 19 électrodes EEG, pour le tracé topographique.

    Les positions viennent du montage standard_1020. Les noms de CH_NAMES sont
    en nomenclature ancienne (T3/T4) alors que le montage MNE utilise la
    nomenclature moderne (T7/T8) : la correspondance est explicitée ici plutôt
    que laissée au hasard d'un appariement partiel silencieux.
    """
    old_to_new = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    ch_names = [old_to_new.get(ch, ch) for ch in CH_NAMES[:N_EEG]]

    info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False)
    return info


def significance_mask(d, acc_mean: np.ndarray, alpha: float,
                      correction: str) -> np.ndarray:
    """Masque booléen des électrodes significatives.

    correction='none'    : p-value par électrode, telle que stockée dans le
                           .npz, comparée à alpha. Non corrigée.
    correction='maxstat' : chaque accuracy est comparée au quantile (1-alpha)
                           de la distribution du MAXIMUM sur les 19 électrodes.
                           Contrôle le FWER sur la carte. C'est le choix
                           d'Arthur (visu_topomap.py, MAXSTAT_ELEC = True).
    """
    if correction == "none":
        if "pvals" not in d:
            return np.zeros(len(acc_mean), dtype=bool)
        return np.asarray(d["pvals"]) < alpha

    # maxstat : loi nulle du maximum inter-électrodes
    if "perm_accs" not in d:
        return np.zeros(len(acc_mean), dtype=bool)
    perm = np.asarray(d["perm_accs"])          # (n_perm, n_elec)
    null_max = perm.max(axis=1)                # (n_perm,)
    ind = max(1, int(alpha * len(null_max)))
    threshold = np.sort(null_max)[-ind]
    return acc_mean > threshold


def main() -> None:
    args = parse_args()
    info = make_info()
    bands = list(FREQ_DICT)

    print(f"=== topomaps {args.feature_family} "
          f"(schéma {args.perm_scheme}, correction {args.correction}) ===")

    # Chargement complet avant tracé : l'échelle de couleur doit être commune
    # à toute la grille pour que les cartes soient comparables entre elles.
    data = {}
    for band in bands:
        for state in STATE_LIST:
            key = f"{args.feature_family}_{band}"
            path = result_path(args.save_path, key, state, args.perm_scheme)
            if not path.exists():
                print(f"  absent : {path.name}")
                continue
            d = np.load(path, allow_pickle=True)
            acc = np.asarray(d["acc_mean"])
            if acc.ndim != 1:
                print(f"  ignoré (pas un vecteur) : {path.name}")
                continue
            data[(band, state)] = (acc * 100, significance_mask(d, acc, args.alpha,
                                                                args.correction))

    if not data:
        raise RuntimeError(
            f"Aucun résultat trouvé pour la famille {args.feature_family} "
            f"(schéma {args.perm_scheme}) dans {args.save_path / 'results'}."
        )

    all_acc = np.concatenate([v[0] for v in data.values()])
    vmin = args.vmin if args.vmin is not None else float(all_acc.min())
    vmax = args.vmax if args.vmax is not None else float(all_acc.max())

    fig, axes = plt.subplots(len(bands), len(STATE_LIST),
                             figsize=(2.2 * len(STATE_LIST), 2.2 * len(bands)))
    axes = np.atleast_2d(axes)

    im = None
    for r, band in enumerate(bands):
        for c, state in enumerate(STATE_LIST):
            ax = axes[r, c]
            if (band, state) not in data:
                ax.axis("off")
                continue
            acc, mask = data[(band, state)]
            im, _ = mne.viz.plot_topomap(
                acc, info, axes=ax, show=False, cmap="viridis",
                vlim=(vmin, vmax), mask=mask,
                mask_params=dict(marker="o", markerfacecolor="w",
                                 markeredgecolor="k", markersize=4),
                contours=0,
            )
            if r == 0:
                ax.set_title(state, fontsize=11)
            if c == 0:
                ax.text(-0.25, 0.5, band, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                            pad=0.02, aspect=30)
        cbar.set_label("Decoding accuracy (%)")

    fig.suptitle(
        f"{args.feature_family} — accuracy par électrode "
        f"(perm. {args.perm_scheme}, p < {args.alpha}, corr. {args.correction})",
        fontsize=12,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = (args.out_dir /
           f"topomap_{args.feature_family}_{args.perm_scheme}_"
           f"{args.correction}_p{args.alpha}.png")
    fig.savefig(out, dpi=RESOLUTION, bbox_inches="tight")
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
