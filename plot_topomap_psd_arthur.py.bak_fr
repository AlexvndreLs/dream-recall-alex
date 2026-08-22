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

Note sur la correction des comparaisons multiples : --correction none reproduit
Arthur (p brute par électrode) ; --correction maxstat applique une correction
FWER sur les 19 électrodes d'une même carte, en comparant chaque accuracy au
maximum de la loi nulle sur les électrodes. La seconde est plus conservatrice
et n'est PAS ce que fait la thèse.

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

# Ordre d'affichage des stades comme dans la Fig. 4 d'Arthur : S2, SWS, NREM,
# REM (les NREM groupés puis REM). STATE_LIST de config_v3 suit un autre ordre
# (S2, SWS, REM, NREM) sans logique d'affichage ; on réordonne ici, en ne
# gardant que les stades effectivement présents dans STATE_LIST.
STATES_DISPLAY = [s for s in ["S2", "SWS", "NREM", "REM"] if s in STATE_LIST]


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
    p.add_argument("--correction", choices=["none", "maxstat"], default="none",
                   help="none = p brute par électrode (Arthur), maxstat = "
                        "correction FWER sur les 19 électrodes.")
    p.add_argument("--vmin", type=float, default=None,
                   help="Borne basse de l'échelle de couleur (%%). "
                        "Par défaut : min des données.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Borne haute de l'échelle de couleur (%%).")
    p.add_argument("--coord-file", type=Path, default=None,
                   help="Fichier de coordonnées cartésiennes 3D (une ligne x y z "
                        "par électrode, dans l'ordre de CH_NAMES), pour reproduire "
                        "exactement le montage d'Arthur (coord_cart_new.txt). "
                        "Sans lui : montage MNE standard_1020, positions légèrement "
                        "différentes au bord.")
    p.add_argument("--sphere", type=float, default=0.11,
                   help="Rayon du contour de tête tracé (m). Plus grand que le "
                        "rayon des électrodes de bord, il les fait rentrer dans "
                        "le cercle et empêche la couleur de déborder. Défaut : "
                        "auto (MNE), qui laisse déborder. Essayer 0.11 avec le "
                        "montage d'Arthur (électrodes à ~0.095).")
    return p.parse_args()


def result_path(save_path: Path, key: str, state: str, scheme: str) -> Path:
    """Chemin du .npz de résultats pour un couple (feature, stade)."""
    suffix = "_epochperm" if scheme == "epoch" else ""
    return save_path / "results" / f"{key}_{state}{suffix}.npz"


def make_info(coord_file: Path | None = None) -> mne.Info:
    """Info MNE portant les 19 électrodes EEG, pour le tracé topographique.

    Si coord_file est fourni : positions lues depuis ce fichier (coordonnées
    cartésiennes 3D x y z, une ligne par électrode dans l'ordre de CH_NAMES).
    C'est le montage exact d'Arthur (coord_cart_new.txt), qui place les
    électrodes de bord (Fp1/Fp2/O1/O2) au même endroit que sa Fig. 4 et évite
    que les étoiles mordent le tracé du crâne.

    Sinon : montage MNE standard_1020, avec correspondance ancienne->moderne
    (T3->T7 etc.). Positions proches mais légèrement différentes au bord.
    """
    if coord_file is not None:
        coords = np.loadtxt(coord_file)
        if coords.shape != (N_EEG, 3):
            raise ValueError(
                f"{coord_file} : attendu ({N_EEG}, 3), lu {coords.shape}. "
                f"Le fichier doit avoir une ligne x y z par électrode."
            )
        ch_names = list(CH_NAMES[:N_EEG])
        # Conversion de repère. Le fichier d'Arthur utilise x=avant (Fp:+x,
        # O:-x), y=gauche (C3:+y, C4:-y), z=haut. MNE "head" attend x=droite,
        # y=avant, z=haut. On permute donc : x_mne = -y_arthur, y_mne = x_arthur.
        # Sans cette conversion la carte est pivotée de 90 degrés, erreur
        # silencieuse (la topographie reste plausible mais fausse).
        xa, ya, za = coords[:, 0], coords[:, 1], coords[:, 2]
        coords_mne = np.column_stack([-ya, xa, za])
        # Mise à l'échelle d'une tête réelle (~0.095 m de rayon) : les
        # coordonnées d'Arthur sont sur la sphère unité.
        pos = {ch: coords_mne[i] * 0.095 for i, ch in enumerate(ch_names)}
        montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head")
        info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
        info.set_montage(montage)
        return info

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
                           .npz, comparée à alpha. C'est le choix d'Arthur.
    correction='maxstat' : chaque accuracy est comparée au quantile (1-alpha)
                           de la distribution du MAXIMUM sur les 19 électrodes.
                           Contrôle le FWER sur la carte, plus conservateur.
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
    info = make_info(args.coord_file)
    bands = list(FREQ_DICT)

    print(f"=== topomaps {args.feature_family} "
          f"(schéma {args.perm_scheme}, correction {args.correction}) ===")

    # Chargement complet avant tracé : l'échelle de couleur doit être commune
    # à toute la grille pour que les cartes soient comparables entre elles.
    data = {}
    for band in bands:
        for state in STATES_DISPLAY:
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

    fig, axes = plt.subplots(len(bands), len(STATES_DISPLAY),
                             figsize=(2.2 * len(STATES_DISPLAY), 2.2 * len(bands)))
    axes = np.atleast_2d(axes)

    im = None
    for r, band in enumerate(bands):
        for c, state in enumerate(STATES_DISPLAY):
            ax = axes[r, c]
            if (band, state) not in data:
                ax.axis("off")
                continue
            acc, mask = data[(band, state)]
            im, _ = mne.viz.plot_topomap(
                acc, info, axes=ax, show=False, cmap="viridis",
                vlim=(vmin, vmax), mask=mask,
                mask_params=dict(marker="*", markerfacecolor="white",
                                 markeredgecolor="white", markersize=7),
                contours=0, sphere=args.sphere,
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
        f"{args.feature_family}, accuracy par électrode "
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