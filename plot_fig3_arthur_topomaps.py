"""Plot Fig. 3 VERSION ARTHUR (these chap.1) : 4 colonnes de topomaps, en S2.

Reproduit la mise en forme EXACTE de la Fig.3 d'Arthur (grille bandes x 4 colonnes) :
  Colonne 1 : PSD HR   (topomap, puissance moyenne par bande par electrode)
  Colonne 2 : PSD LR   (topomap)
  Colonne 3 : corrected T-values (topomap, nos t-values recompute + etoiles p<0.001)
  Colonne 4 : Decoding accuracies (topomap LDA par electrode + etoiles p<0.001)
5 lignes = 5 bandes (delta, theta, alpha, sigma, beta).

Sources :
  - PSD HR/LR : features psd_{band}_s{XX}_{state}.npz (moyenne epochs puis sujets par
    groupe). PAS de recompute : c'est la puissance par bande/electrode deja extraite,
    exactement ce qu'il faut pour une topomap par bande (identique a Arthur).
  - T-values : fig3_ttest_{state}.npz (recompute_ttest_fig3.py).
  - LDA : {results}/psd_{band}_{state}{suffix}.npz (classify.py), optionnel.

Complement de plot_fig3_arthur.py (qui, lui, fait la version courbe PSD). Les deux
coexistent : ce script = mise en forme fidele Arthur, l'autre = version courbe spectre.

Usage
-----
    python plot_fig3_arthur_topomaps.py \\
        --save-path /scratch/alouis/dream_features_noica_1000hz \\
        --in-dir    /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute \\
        --results   /scratch/alouis/dream_features_noica_1000hz/results \\
        --coord-file coord_cart_new.txt \\
        --state S2 --out fig3_arthur_S2.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from config_v3 import (
    FREQ_DICT, CH_NAMES, N_EEG,
    SUBJECT_LIST_ORDERED, SUBJECT_LABELS, CLASSIFICATION_GROUPS,
)
from utils import load_atomic

BANDS = list(FREQ_DICT)
BAND_LABELS = {b: b.capitalize() for b in BANDS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Dossier des features psd_{band} (pour les topomaps PSD).")
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Dossier fig3_ttest_{state}.npz.")
    p.add_argument("--results", type=Path, default=None,
                   help="Dossier results/ classify.py (LDA), optionnel.")
    p.add_argument("--coord-file", type=Path, default=None)
    p.add_argument("--state", type=str, default="S2")
    p.add_argument("--suffix", type=str, default="_epochperm")
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--out", type=Path, default=Path("fig3_arthur.png"))
    return p.parse_args()


def make_info(coord_file):
    """Info MNE 19 electrodes. REPRIS de plot_topomap_psd_arthur.py."""
    if coord_file is not None:
        coords = np.loadtxt(coord_file)
        if coords.shape != (N_EEG, 3):
            raise ValueError(f"{coord_file}: attendu ({N_EEG},3), lu {coords.shape}")
        ch_names = list(CH_NAMES[:N_EEG])
        xa, ya, za = coords[:, 0], coords[:, 1], coords[:, 2]
        coords_mne = np.column_stack([-ya, xa, za])
        pos = {ch: coords_mne[i] * 0.095 for i, ch in enumerate(ch_names)}
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


def psd_topomap_data(save_path, state, drop_subject_10=True):
    """Retourne psd_hr[band], psd_lr[band] : (19,) puissance NORMALISEE par electrode.

    Reproduit prepare_recallers d'Arthur (visu_gen_fig1.py) :
      - moyenne sur epochs puis sur sujets de chaque groupe,
      - NORMALISATION par le max (HR /= HR.max()) -> echelle 0..1 par bande, ce qui
        rend chaque bande lisible (sinon la puissance brute ~1e-10 ecrase tout),
      - EXCLUSION du sujet 10 (artefact FC2) : Arthur le retire aussi pour ce plot PSD
        (pas seulement pour le ttest). drop_subject_10=True par defaut.
    """
    stages = CLASSIFICATION_GROUPS[state]
    per_band_hr = {b: [] for b in BANDS}
    per_band_lr = {b: [] for b in BANDS}
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        if drop_subject_10 and str(sub_id) == "10":
            continue
        for b in BANDS:
            parts = [a for s in stages
                     if (a := load_atomic(save_path, f"psd_{b}", sub_id, s)) is not None]
            if not parts:
                continue
            arr = np.concatenate(parts, axis=0)     # (n_epochs, 19)
            mean_elec = arr.mean(axis=0)            # (19,) moyenne sur epochs
            (per_band_hr if label == 1 else per_band_lr)[b].append(mean_elec)
    psd_hr, psd_lr = {}, {}
    for b in BANDS:
        hr = np.mean(per_band_hr[b], axis=0)
        lr = np.mean(per_band_lr[b], axis=0)
        # normalisation par le max (comme Arthur : HR /= HR.max())
        psd_hr[b] = hr / hr.max()
        psd_lr[b] = lr / lr.max()
    return psd_hr, psd_lr


def main():
    args = parse_args()
    info = make_info(args.coord_file)

    # --- charge les 4 sources
    psd_hr, psd_lr = psd_topomap_data(args.save_path, args.state)

    ttest_path = args.in_dir / f"fig3_ttest_{args.state}.npz"
    if not ttest_path.exists():
        raise FileNotFoundError(ttest_path)
    dt = np.load(ttest_path)
    tvals = dt["tvals"]     # (5,19)
    pvals = dt["pvals"]     # (5,19)

    lda = {}
    if args.results is not None:
        for b in BANDS:
            path = args.results / f"psd_{b}_{args.state}{args.suffix}.npz"
            if path.exists():
                dd = np.load(path, allow_pickle=True)
                if "acc_mean" in dd:
                    lda[b] = dd

    # echelle LDA : Arthur fixe cbarlim=[50,60] en dur (visu_topomap.py ligne 174)
    lda_vmin, lda_vmax = 50.0, 60.0

    # --- echelles. PSD normalisee par max -> 0..1, mais on laisse chaque bande a son
    # echelle (vlim None) pour un rendu type Arthur (chaque topomap lisible).
    # Les t-values sont z-scorees par bande a l'affichage (cf boucle, comme Arthur).
    ch_names_disp = list(CH_NAMES[:N_EEG])

    ncol = 4
    fig, axes = plt.subplots(len(BANDS), ncol, figsize=(11, 2.4 * len(BANDS)))
    col_titles = ["PSD HR", "PSD LR", "corrected T-values", "Decoding Accuracies (%)"]

    ims = [None] * ncol
    for r, b in enumerate(BANDS):
        # col 0 : PSD HR (normalisee, avec noms d'electrodes)
        im0, _ = mne.viz.plot_topomap(psd_hr[b], info, axes=axes[r, 0], show=False,
                                      cmap="jet", extrapolate="head", sphere=0.11,
                                      contours=0, names=ch_names_disp)
        ims[0] = im0
        # col 1 : PSD LR
        im1, _ = mne.viz.plot_topomap(psd_lr[b], info, axes=axes[r, 1], show=False,
                                      cmap="jet", extrapolate="head", sphere=0.11,
                                      contours=0, names=ch_names_disp)
        ims[1] = im1
        # col 2 : T-values + etoiles.
        # AFFICHAGE fidele a Arthur (visu_topomap.py) :
        #  - t_values = zscore(t) ligne 114 (z-score par bande),
        #  - cbarlim = [min, max] REELS ligne 168 (echelle ASYMETRIQUE, pas +-symetrique),
        #  - cmap viridis ligne 166 (pas RdBu),
        #  - mask = p-values < alpha ligne 128 (etoiles depuis les p, pas les t affiches).
        mask_t = pvals[r] < args.alpha
        tvals_z = (tvals[r] - tvals[r].mean()) / tvals[r].std()
        im2, _ = mne.viz.plot_topomap(tvals_z, info, axes=axes[r, 2], show=False,
                                      cmap="viridis",
                                      vlim=(tvals_z.min(), tvals_z.max()),
                                      extrapolate="head", sphere=0.11, contours=0,
                                      mask=mask_t,
                                      mask_params=dict(marker="*", markerfacecolor="w",
                                                       markeredgecolor="k", markersize=9,
                                                       markeredgewidth=0.3, linewidth=0))
        ims[2] = im2
        # col 3 : LDA
        if b in lda:
            dd = lda[b]
            acc = np.asarray(dd["acc_mean"]) * 100
            if "perm_accs" in dd:
                perm = np.asarray(dd["perm_accs"]); nm = perm.max(axis=1)
                ind = max(1, int(args.alpha * len(nm))); thr = np.sort(nm)[-ind]
                mask_l = np.asarray(dd["acc_mean"]) > thr
            elif "pvals" in dd:
                mask_l = np.asarray(dd["pvals"]) < args.alpha
            else:
                mask_l = np.zeros(len(acc), dtype=bool)
            # echelle LDA resserree autour de la chance (comme Arthur : 50-60%),
            # calculee sur l'ensemble des bandes pour comparabilite
            im3, _ = mne.viz.plot_topomap(acc, info, axes=axes[r, 3], show=False,
                                          cmap="viridis", extrapolate="head",
                                          vlim=(lda_vmin, lda_vmax),
                                          sphere=0.11, contours=0, mask=mask_l,
                                          mask_params=dict(marker="*",
                                                           markerfacecolor="w",
                                                           markeredgecolor="k",
                                                           markersize=9, markeredgewidth=0.3, linewidth=0))
            ims[3] = im3
        else:
            axes[r, 3].axis("off")

        axes[r, 0].set_ylabel(BAND_LABELS[b], fontsize=12, rotation=90, labelpad=15)

    for c in range(ncol):
        axes[0, c].set_title(col_titles[c], fontsize=11)

    # colorbars sous chaque colonne
    for c, label in zip(range(ncol), ["PSD (norm.)", "PSD (norm.)", "t-value (z)", "Acc (%)"]):
        if ims[c] is not None:
            cax = fig.add_axes([0.13 + c * 0.205, 0.06, 0.13, 0.012])
            fig.colorbar(ims[c], cax=cax, orientation="horizontal", label=label)

    fig.suptitle(f"Fig. 3 (Arthur chap.1), HR vs LR, {args.state}  "
                 f"(stars: p<{args.alpha}, maxstat)", fontsize=12, y=0.99)
    fig.subplots_adjust(bottom=0.11, hspace=0.15, wspace=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {args.out}")


if __name__ == "__main__":
    main()