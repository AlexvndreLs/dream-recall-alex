#!/usr/bin/env python3
"""Demonstration figure FFX vs RFX for Fig. 3 (Chap. 1 Arthur), in S2.

A single figure, 4 blocks in a 2x2 grid:

                    t-values column        accuracy column (LDA)
    FFX row (top)   [ t FFX ]              [ acc FFX ]
    RFX row (bottom)[ t RFX ]              [ acc RFX ]

Each block = vertical grid of 5 frequency bands (delta -> beta), one topomap per band.
Goal: show that the stars present in FFX (epoch-level permutation, inflated n)
DISAPPEAR in RFX (subject-level permutation), both for the t-test and for
decoding, while the underlying values are identical (accuracy does not change
between FFX and RFX, only the null distribution changes).

Scales:
  - accuracy : shared 50-60 (IDENTICAL colors top/bottom, only stars move).
  - t-values : z-score per band (each map readable; FFX and RFX t-values have
    very different magnitudes, a shared scale would crush RFX).

Sources:
  - t FFX  : {ttest_ffx}/fig3_ttest_{state}.npz
  - t RFX  : {ttest_rfx}/fig3_ttest_{state}.npz
  - acc FFX: {results}/psd_{band}_{state}_epochperm.npz
  - acc RFX: {results}/psd_{band}_{state}.npz

Purely visual: no recalculation, reads existing .npz files.

Usage
-----
    python plot_ffx_vs_rfx_2x2.py \
        --ttest-ffx /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute_correct \
        --ttest-rfx /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute_arthurRFX \
        --results   /scratch/alouis/dream_features_noica_1000hz/results \
        --coord-file coord_cart_new.txt \
        --state S2 --alpha-ffx 0.001 --alpha-rfx 0.05 \
        --out ~/dream-recall-alex/figures/fig3_ffx_vs_rfx_S2.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from config_v3 import FREQ_DICT, CH_NAMES, N_EEG

BANDS = list(FREQ_DICT)
BAND_LABELS = {b: b.capitalize() for b in BANDS}

LDA_VMIN, LDA_VMAX = 50.0, 60.0  # Shared accuracy scale (like Arthur)

STAR = dict(marker="*", markerfacecolor="w", markeredgecolor="k",
            markersize=9, markeredgewidth=0.3, linewidth=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ttest-ffx", type=Path, required=True,
                   help="Folder of fig3_ttest_{state}.npz calculated at --level epoch.")
    p.add_argument("--ttest-rfx", type=Path, required=True,
                   help="Folder of fig3_ttest_{state}.npz calculated at --level subject.")
    p.add_argument("--results", type=Path, required=True,
                   help="Results folder containing psd_{band}_{state}.npz (RFX) and "
                        "psd_{band}_{state}_epochperm.npz (FFX).")
    p.add_argument("--coord-file", type=Path, default=None)
    p.add_argument("--state", type=str, default="S2")
    p.add_argument("--alpha-ffx", type=float, default=0.001,
                   help="Threshold for FFX columns (epoch perm). Default 0.001 "
                        "(Arthur's threshold: inflated n, strict setting).")
    p.add_argument("--alpha-rfx", type=float, default=0.05,
                   help="Threshold for RFX columns (subject perm). Default 0.05 "
                        "(standard threshold: honest n at subject level).")
    p.add_argument("--out", type=Path, default=Path("fig3_ffx_vs_rfx.png"))
    return p.parse_args()


def make_info(coord_file):
    """MNE Info 19 electrodes. Identical to other scripts (Arthur montage)."""
    if coord_file is not None:
        coords = np.loadtxt(coord_file)
        if coords.shape != (N_EEG, 3):
            raise ValueError(f"{coord_file}: expected ({N_EEG},3), read {coords.shape}")
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


def load_ttest(folder, state):
    """Returns tvals (5,19) and pvals (5,19) from a fig3_ttest folder."""
    path = folder / f"fig3_ttest_{state}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path)
    return d["tvals"], d["pvals"]


def load_acc(results, band, state, scheme, alpha):
    """Returns (acc %, star mask) for the decoding of a frequency band.

    scheme='ffx'    -> file psd_{band}_{state}_epochperm.npz
    scheme='rfx'    -> file psd_{band}_{state}.npz
    The maxstat mask: each electrode compared to the (1-alpha) quantile of the
    null distribution of MAX across electrodes (perm_accs (n_perm, 19)).
    """
    suffix = "_epochperm" if scheme == "ffx" else ""
    path = results / f"psd_{band}_{state}{suffix}.npz"
    if not path.exists():
        return None, None
    d = np.load(path, allow_pickle=True)
    acc = np.asarray(d["acc_mean"])
    if acc.ndim != 1:
        return None, None
    mask = np.zeros(len(acc), dtype=bool)
    if "perm_accs" in d:
        perm = np.asarray(d["perm_accs"])
        if perm.ndim == 2:
            null_max = perm.max(axis=1)
            ind = max(1, int(alpha * len(null_max)))
            thr = np.sort(null_max)[-ind]
            mask = acc > thr
    return acc * 100, mask


def main():
    args = parse_args()
    info = make_info(args.coord_file)

    # --- load t-values FFX / RFX
    t_ffx, p_ffx = load_ttest(args.ttest_ffx, args.state)
    t_rfx, p_rfx = load_ttest(args.ttest_rfx, args.state)

    # --- load accuracy FFX / RFX per band
    acc_ffx = {b: load_acc(args.results, b, args.state, "ffx", args.alpha_ffx) for b in BANDS}
    acc_rfx = {b: load_acc(args.results, b, args.state, "rfx", args.alpha_rfx) for b in BANDS}

    n_bands = len(BANDS)
    # 5 rows (bands) x 4 columns: [t FFX | acc FFX | t RFX | acc RFX]
    fig, axes = plt.subplots(n_bands, 4, figsize=(11, 2.3 * n_bands))
    axes = np.atleast_2d(axes)

    col_titles = [f"t-values (FFX,\nepoch perm, p<{args.alpha_ffx})",
                  f"Decoding (FFX,\nepoch perm, p<{args.alpha_ffx})",
                  f"t-values (RFX,\nsubject perm, p<{args.alpha_rfx})",
                  f"Decoding (RFX,\nsubject perm, p<{args.alpha_rfx})"]

    im_t = im_acc = None
    for r, b in enumerate(BANDS):
        # col 0: t-values FFX (z-score per band)
        sd = t_ffx[r].std()
        tz = (t_ffx[r] - t_ffx[r].mean()) / sd if sd > 0 else np.zeros_like(t_ffx[r])
        im_t, _ = mne.viz.plot_topomap(
            tz, info, axes=axes[r, 0], show=False, cmap="viridis",
            vlim=(tz.min(), tz.max()) if sd > 0 else (-1, 1),
            extrapolate="head", sphere=0.11, contours=0,
            mask=p_ffx[r] < args.alpha_ffx, mask_params=STAR)

        # col 1: accuracy FFX (shared scale 50-60)
        acc, mask = acc_ffx[b]
        if acc is not None:
            im_acc, _ = mne.viz.plot_topomap(
                acc, info, axes=axes[r, 1], show=False, cmap="viridis",
                vlim=(LDA_VMIN, LDA_VMAX), extrapolate="head", sphere=0.11,
                contours=0, mask=mask, mask_params=STAR)
        else:
            axes[r, 1].axis("off")

        # col 2: t-values RFX (z-score per band)
        sd = t_rfx[r].std()
        tz = (t_rfx[r] - t_rfx[r].mean()) / sd if sd > 0 else np.zeros_like(t_rfx[r])
        mne.viz.plot_topomap(
            tz, info, axes=axes[r, 2], show=False, cmap="viridis",
            vlim=(tz.min(), tz.max()) if sd > 0 else (-1, 1),
            extrapolate="head", sphere=0.11, contours=0,
            mask=p_rfx[r] < args.alpha_rfx, mask_params=STAR)

        # col 3: accuracy RFX (same scale 50-60: colors identical to col 1)
        acc, mask = acc_rfx[b]
        if acc is not None:
            mne.viz.plot_topomap(
                acc, info, axes=axes[r, 3], show=False, cmap="viridis",
                vlim=(LDA_VMIN, LDA_VMAX), extrapolate="head", sphere=0.11,
                contours=0, mask=mask, mask_params=STAR)
        else:
            axes[r, 3].axis("off")

        axes[r, 0].set_ylabel(BAND_LABELS[b], fontsize=12, rotation=90, labelpad=15)

    for c in range(4):
        axes[0, c].set_title(col_titles[c], fontsize=10)

    # visual separator between FFX block (col 0-1) and RFX block (col 2-3)
    fig.subplots_adjust(bottom=0.10, top=0.90, wspace=0.05, hspace=0.15)
    # colorbars
    if im_t is not None:
        cax = fig.add_axes([0.13, 0.05, 0.20, 0.012])
        fig.colorbar(im_t, cax=cax, orientation="horizontal", label="t-value (z, per band)")
    if im_acc is not None:
        cax = fig.add_axes([0.58, 0.05, 0.20, 0.012])
        fig.colorbar(im_acc, cax=cax, orientation="horizontal", label="Decoding accuracy (%)")

    fig.suptitle(
        f"Fig. 3, {args.state}: EPOCH level permutation (FFX, p<{args.alpha_ffx}) "
        f"vs SUBJECT level (RFX, p<{args.alpha_rfx})\n"
        f"decoding: IDENTICAL accuracy (only the null distribution changes); t-values: "
        f"recomputed by level. Maxstat electrodes. Stars drop in RFX.",
        fontsize=11, y=0.985)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {args.out}")


if __name__ == "__main__":
    main()