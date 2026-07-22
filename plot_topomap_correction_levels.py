"""Comparaison de 3 niveaux de correction des comparaisons multiples sur UNE
topomap (une bande, un stade). Figure méthodologique, pas de résultat.

But : montrer visuellement comment le nombre d'électrodes déclarées significatives
s'effondre à mesure que la correction devient rigoureuse. Trois colonnes, même
carte d'accuracy, trois masques de significativité différents :

  1. brute     : p par électrode telle quelle, AUCUNE correction. Sur 19
                 électrodes on attend ~1 faux positif par pur hasard à p<0.05,
                 sur 95 (toute la famille) ~5. Cette colonne N'EST PAS un
                 résultat, c'est le repoussoir : ce qu'on obtiendrait en ne
                 corrigeant pas.
  2. max-stat  : correction sur les 19 électrodes de cette carte seulement
     (Arthur)   (visu_topomap.py, MAXSTAT_ELEC = True). Contrôle le FWER par
                 carte, pas entre bandes.
  3. pooled    : correction sur toute la famille (5 bandes x 19 = 95 tests),
                 lue depuis compute_maxstat_correction.py. La plus stricte, celle
                 des figures propres.

Les p-values brute et max-stat sont recalculées ici depuis le .npz de la carte ;
la pooled est lue depuis le .npz corrigé pour rester cohérente avec le reste du
pipeline (une seule définition du pooling).

Usage :
    python plot_topomap_correction_levels.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /scratch/alouis/dream-recall-alex/plot_overlap \
        --band sigma --state S2 --family psd_classic \
        --coord-file coord_cart_new.txt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from config_v3 import CH_NAMES, N_EEG
from plot_common import RESOLUTION, load_result

# familles vectorielles supportées : préfixe de clé par famille
FAMILY_PREFIX = {"psd_classic": "psd", "psd_osc": "psd_osc"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--corrected-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--band", default="sigma")
    p.add_argument("--state", default="S2")
    p.add_argument("--family", default="psd_classic", choices=list(FAMILY_PREFIX))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--coord-file", type=Path, default=None)
    p.add_argument("--sphere", type=float, default=0.11)
    return p.parse_args()


def make_info(coord_file: Path | None) -> mne.Info:
    """Montage 19 EEG. coord_file = montage exact d'Arthur (conversion d'axes
    x=avant/y=gauche -> x=droite/y=avant), sinon standard_1020."""
    if coord_file is not None:
        coords = np.loadtxt(coord_file)
        if coords.shape != (N_EEG, 3):
            raise ValueError(f"{coord_file}: attendu ({N_EEG}, 3), lu {coords.shape}")
        ch_names = list(CH_NAMES[:N_EEG])
        xa, ya, za = coords[:, 0], coords[:, 1], coords[:, 2]
        pos = {ch: np.array([-ya[i], xa[i], za[i]]) * 0.095
               for i, ch in enumerate(ch_names)}
        montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head")
        info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
        info.set_montage(montage)
        return info
    old_to_new = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    ch_names = [old_to_new.get(ch, ch) for ch in CH_NAMES[:N_EEG]]
    info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False)
    return info


def mask_raw(d, alpha: float) -> np.ndarray:
    """p brute par électrode, non corrigée."""
    return np.asarray(d["pvals"]) < alpha


def mask_maxstat_elec(d, alpha: float) -> np.ndarray:
    """Correction max-stat sur les 19 électrodes de cette carte (Arthur).

    La loi nulle est le maximum des accuracies permutées sur les électrodes ;
    on compare chaque accuracy réelle au quantile (1-alpha) de cette loi.
    """
    perm = np.asarray(d["perm_accs"])          # (n_perm, 19)
    acc = np.asarray(d["acc_mean"])            # (19,)
    null_max = perm.max(axis=1)                # (n_perm,)
    thr = np.quantile(null_max, 1 - alpha)
    return acc > thr


def mask_pooled(corrected_path: Path, family: str, band: str, state: str,
                ch_names: list[str], alpha: float) -> np.ndarray:
    """p pooled sur toute la famille, lue depuis le .npz corrigé.

    test_labels a la forme "psd_sigma/Fz" : on filtre sur la bande demandée.
    """
    prefix = FAMILY_PREFIX[family]
    key = f"{prefix}_{band}"
    p = corrected_path / f"{family}_{state}_maxstat.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} absent. Lancer compute_maxstat_correction.py --family-name {family} "
            f"d'abord."
        )
    c = np.load(p, allow_pickle=True)
    labels = [str(x) for x in c["test_labels"]]
    pvals = {lab: pv for lab, pv in zip(labels, c["pvals_corrected"])}
    return np.array([pvals.get(f"{key}/{ch}", 1.0) < alpha for ch in ch_names])


def main() -> None:
    args = parse_args()
    info = make_info(args.coord_file)
    prefix = FAMILY_PREFIX[args.family]
    key = f"{prefix}_{args.band}"

    d = load_result(args.save_path, key, args.state)
    if d is None:
        raise FileNotFoundError(f"{key}_{args.state}.npz absent dans {args.save_path}")
    acc = np.asarray(d["acc_mean"]) * 100
    ch_names = [str(c) for c in d["ch_names"]]

    masks = [
        ("Sans correction (p brute)", mask_raw(d, args.alpha)),
        ("Max-stat 19 électrodes (Arthur)", mask_maxstat_elec(d, args.alpha)),
        ("Max-stat pooled (95 tests)",
         mask_pooled(args.corrected_path, args.family, args.band, args.state,
                     ch_names, args.alpha)),
    ]

    vmin = args.vmin if args.vmin is not None else float(acc.min())
    vmax = args.vmax if args.vmax is not None else float(acc.max())

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    im = None
    for ax, (title, mask) in zip(axes, masks):
        im, _ = mne.viz.plot_topomap(
            acc, info, axes=ax, show=False, cmap="viridis",
            vlim=(vmin, vmax), mask=mask,
            mask_params=dict(marker="*", markerfacecolor="white",
                             markeredgecolor="white", markersize=9),
            contours=0, sphere=args.sphere,
        )
        ax.set_title(f"{title}\n{int(mask.sum())}/19 électrodes", fontsize=10)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.03,
                            aspect=25)
        cbar.set_label("Decoding accuracy (%)")

    fig.suptitle(
        f"Effet de la correction des comparaisons multiples, {args.band} / {args.state} "
        f"(perm. sujet, p < {args.alpha})\n"
        f"figure méthodologique : la colonne de gauche N'EST PAS un résultat "
        f"(faux positifs attendus sans correction)",
        fontsize=11,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = (args.out_dir /
           f"topomap_correction_levels_{args.band}_{args.state}_p{args.alpha}.png")
    fig.savefig(out, dpi=RESOLUTION, bbox_inches="tight")
    plt.close(fig)
    for title, mask in masks:
        print(f"  {title}: {int(mask.sum())}/19")
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
