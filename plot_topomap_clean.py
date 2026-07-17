"""Topomaps des accuracies de décodage par électrode.

Version "propre" : schéma de permutation SUBJECT (RFX), correction max-stat
POOLED sur toute la famille (5 bandes x 19 électrodes = 95 tests par stade).
À distinguer de plot_topomap_psd_arthur.py, qui réplique le chapitre 1 (schéma
epoch, max-stat sur 19 électrodes seulement, échelle figée [50,60]).

Chaque feature vectorielle est classifiée une électrode à la fois (LDA
univarié) : acc_mean est un vecteur de 19 accuracies. La grille bandes x stades
donne la distribution spatiale du pouvoir décodant.

Deux écarts assumés vs la version Arthur :
  - extrapolate="head" : MNE extrapole par défaut bien au-delà des capteurs et
    colore des zones hors du scalp qui ne correspondent à aucune donnée (visible
    chez Arthur : du jaune >60% en occipital là où la meilleure électrode
    plafonne à 57%). "head" borne l'interpolation au contour du crâne. Le mode
    "local", plus strict encore, produit un contour hexagonal avec seulement 19
    électrodes : disponible via --extrapolate local, pas retenu par défaut.
  - échelle calculée sur les données plutôt que figée à [50,60] : évite de
    saturer les cartes qui sortent de cette plage.

Usage :
    python plot_topomap_clean.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /scratch/alouis/dream-recall-alex/plot_overlap \
        --family psd_classic --alpha 0.05
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from config_v3 import CH_NAMES, N_EEG
from plot_common import (
    BANDS,
    FAMILY_KEYS,
    RESOLUTION,
    STATES_ORDERED,
    load_maxstat,
    load_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--corrected-path", type=Path, default=None,
                   help="Dossier des .npz maxstat. Sans lui, aucune électrode marquée.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--family", default="psd_classic",
                   choices=["psd_classic", "psd_osc"],
                   help="psd_classic = puissance brute (Arthur), psd_osc = ratio "
                        "puissance/aperiodic (extension FOOOF).")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--vmin", type=float, default=None,
                   help="Borne basse de l'échelle (%%). Défaut : percentile 2 des données.")
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--extrapolate", default="head", choices=["local", "head", "box"],
                   help="head = interpolation bornée au crâne (défaut, forme ronde). "
                        "local = restreint au nuage de capteurs, mais produit un "
                        "contour hexagonal disgracieux avec un montage 19 électrodes. "
                        "box = défaut MNE, déborde largement hors du scalp.")
    return p.parse_args()


def make_info() -> mne.Info:
    """Info MNE portant les 19 électrodes EEG, pour le tracé topographique.

    CH_NAMES est en nomenclature ancienne (T3/T4) alors que standard_1020 utilise
    la moderne (T7/T8) : la correspondance est explicitée plutôt que laissée à un
    appariement partiel silencieux.
    """
    old_to_new = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    ch_names = [old_to_new.get(ch, ch) for ch in CH_NAMES[:N_EEG]]
    info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False)
    return info


def main() -> None:
    args = parse_args()
    keys = FAMILY_KEYS[args.family]
    info = make_info()

    print(f"=== topomaps {args.family} (subject, max-stat pooled, p < {args.alpha}) ===")

    # Chargement complet avant tracé : l'échelle doit être commune à la grille
    # pour que les cartes soient comparables entre elles.
    data = {}
    for state in STATES_ORDERED:
        pvals = load_maxstat(args.corrected_path, args.family, state) \
            if args.corrected_path else None

        for band, key in zip(BANDS, keys):
            d = load_result(args.save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                continue
            acc = np.asarray(d["acc_mean"])
            if acc.ndim != 1:
                print(f"  ignoré (pas un vecteur) : {key}_{state}.npz")
                continue

            if pvals is None:
                mask = np.zeros(len(acc), dtype=bool)
            else:
                # test_labels de compute_maxstat_correction.py : "psd_sigma/Fp2"
                ch_names = [str(c) for c in d["ch_names"]]
                mask = np.array([pvals.get(f"{key}/{ch}", 1.0) < args.alpha
                                 for ch in ch_names])
            data[(band, state)] = (acc * 100, mask)

    if not data:
        raise RuntimeError(f"Aucun résultat pour {args.family} dans {args.save_path/'results'}")

    all_acc = np.concatenate([v[0] for v in data.values()])
    # Percentiles plutôt que min/max : quelques électrodes très basses écrasent
    # sinon toute la dynamique utile de la grille.
    vmin = args.vmin if args.vmin is not None else float(np.percentile(all_acc, 2))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(all_acc, 98))

    fig, axes = plt.subplots(len(BANDS), len(STATES_ORDERED),
                             figsize=(2.3 * len(STATES_ORDERED), 2.3 * len(BANDS)))
    axes = np.atleast_2d(axes)

    im, n_sig = None, 0
    for r, band in enumerate(BANDS):
        for c, state in enumerate(STATES_ORDERED):
            ax = axes[r, c]
            if (band, state) not in data:
                ax.axis("off")
                continue
            acc, mask = data[(band, state)]
            n_sig += int(mask.sum())
            im, _ = mne.viz.plot_topomap(
                acc, info, axes=ax, show=False, cmap="viridis",
                vlim=(vmin, vmax), mask=mask,
                mask_params=dict(marker="*", markerfacecolor="w",
                                 markeredgecolor="w", markersize=9),
                contours=0, extrapolate=args.extrapolate,
            )
            if r == 0:
                ax.set_title(state, fontsize=12)
            if c == 0:
                ax.text(-0.2, 0.5, band, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02,
                            aspect=30)
        cbar.set_label("Decoding accuracy (%)")

    fig.suptitle(
        f"{args.family} — accuracy par électrode, permutation sujet (RFX)\n"
        f"* : p < {args.alpha} après max-stat pooled sur {len(BANDS) * N_EEG} tests",
        fontsize=12,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"topomap_{args.family}_subject_pooled_p{args.alpha}.png"
    fig.savefig(out, dpi=RESOLUTION, bbox_inches="tight")
    plt.close(fig)
    print(f"  {n_sig} électrodes significatives sur {len(data) * N_EEG}")
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
