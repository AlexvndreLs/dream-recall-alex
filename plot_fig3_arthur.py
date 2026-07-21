"""Plot Fig. 3 (these Arthur chap.1) : PSD / T-values / LDA en S2, 3 panneaux.

Consomme les .npz du recompute (separation calcul/visu) :
  - Panneau GAUCHE  (PSD)      : fig3_psd_spectrum_{state}.npz (recompute_psd_spectrum_fig3.py)
  - Panneau MILIEU  (T-values) : fig3_ttest_{state}.npz        (recompute_ttest_fig3.py)
  - Panneau DROITE  (LDA)      : {results}/psd_{band}_{state}{suffix}.npz (classify.py),
                                 topomaps accuracy par electrode + etoiles p<0.001.

Le panneau LDA est optionnel : si les resultats classif sont absents, le script trace
les deux premiers panneaux et laisse le 3e vide (avec un message). Aucune invention.

Conventions topomap REPRISES de plot_topomap_psd_arthur.py (make_info, coord_cart_new,
extrapolate='head', sphere 0.11) pour coherence avec les figures existantes.

Usage
-----
    python plot_fig3_arthur.py \\
        --in-dir  /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute \\
        --results /scratch/alouis/dream_features_noica_1000hz/results \\
        --coord-file coord_cart_new.txt \\
        --state S2 \\
        --out fig3_S2.png

    # sans panneau LDA (juste PSD + t-values) : omettre --results
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

from config_v3 import FREQ_DICT, CH_NAMES, N_EEG

BANDS = list(FREQ_DICT)
BAND_LABELS = {b: b.capitalize() for b in BANDS}
# centres de bande pour placer les colonnes du panneau t-values
BAND_CENTERS = {b: (lo + hi) / 2 for b, (lo, hi) in FREQ_DICT.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Dossier des fig3_*.npz (recompute).")
    p.add_argument("--results", type=Path, default=None,
                   help="Dossier results/ classify.py pour le panneau LDA (optionnel).")
    p.add_argument("--coord-file", type=Path, default=None,
                   help="coord_cart_new.txt (montage Arthur). Sinon standard_1020.")
    p.add_argument("--state", type=str, default="S2")
    p.add_argument("--suffix", type=str, default="_epochperm",
                   help="Suffixe des .npz classif LDA (epoch=_epochperm).")
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--out", type=Path, default=Path("fig3.png"))
    return p.parse_args()


def make_info(coord_file):
    """Info MNE 19 electrodes. REPRIS de plot_topomap_psd_arthur.py (identique)."""
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


def panel_psd(ax, npz_path):
    """Panneau gauche : PSD moyenne HR (rouge) vs LR (bleu) + ruban SEM."""
    d = np.load(npz_path)
    freqs = d["freqs"]
    for grp, color, lab in [("lr", "tab:blue", "Low Recallers"),
                            ("hr", "tab:red", "High Recallers")]:
        mean = d[f"psd_{grp}"]
        sem = d[f"sem_{grp}"]
        # affichage en 10*log10 (dB), standard pour une PSD (cf note E2 recompute :
        # transformation d'affichage laissee au plot, SANS le /(k+1) douteux d'Arthur).
        mean_db = 10 * np.log10(mean)
        # ruban SEM converti approximativement en dB autour de la moyenne
        upper = 10 * np.log10(mean + sem)
        lower = 10 * np.log10(np.clip(mean - sem, 1e-30, None))
        ax.plot(freqs, mean_db, color=color, label=lab, lw=1.6)
        ax.fill_between(freqs, lower, upper, color=color, alpha=0.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(f"PSD averaged across electrodes\n(HR={int(d['n_hr'])}, LR={int(d['n_lr'])})")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(freqs.min(), freqs.max())


def panel_tvalues(fig, gs_col, npz_path, info, alpha):
    """Panneau milieu : topomaps t-values corrigees, une par bande, etoiles p<alpha."""
    d = np.load(npz_path)
    bands = [b.decode() if isinstance(b, bytes) else str(b) for b in d["bands"]]
    tvals = d["tvals"]     # (5, 19)
    pvals = d["pvals"]     # (5, 19)
    vmax = np.abs(tvals).max()
    axes = []
    for i, b in enumerate(bands):
        ax = fig.add_subplot(gs_col[i])
        mask = pvals[i] < alpha
        im, _ = mne.viz.plot_topomap(
            tvals[i], info, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), extrapolate="head", sphere=0.11,
            mask=mask,
            mask_params=dict(marker="*", markerfacecolor="k", markeredgecolor="k",
                             markersize=8, linewidth=0),
            contours=0,
        )
        nsig = int(mask.sum())
        ax.set_title(f"{BAND_LABELS.get(b, b)}  ({nsig}/19)", fontsize=9)
        axes.append((ax, im))
    return axes, vmax


def panel_lda(fig, gs_col, results_dir, state, suffix, info, alpha):
    """Panneau droite : topomaps accuracy LDA par electrode + etoiles p<alpha.

    Optionnel : si les .npz classif sont absents, trace des cases vides.
    """
    axes = []
    accs_all = []
    loaded = {}
    for b in BANDS:
        path = results_dir / f"psd_{b}_{state}{suffix}.npz"
        if path.exists():
            dd = np.load(path, allow_pickle=True)
            if "acc_mean" in dd:
                loaded[b] = dd
                accs_all.append(np.asarray(dd["acc_mean"]) * 100)
    if not loaded:
        # rien : cases vides avec message
        for i in range(len(BANDS)):
            ax = fig.add_subplot(gs_col[i])
            ax.axis("off")
            if i == 0:
                ax.set_title("LDA absent\n(pas de results/)", fontsize=9)
        return axes, None, None
    vmin = min(a.min() for a in accs_all)
    vmax = max(a.max() for a in accs_all)
    for i, b in enumerate(BANDS):
        ax = fig.add_subplot(gs_col[i])
        if b not in loaded:
            ax.axis("off")
            continue
        dd = loaded[b]
        acc = np.asarray(dd["acc_mean"]) * 100
        # masque significativite : maxstat sur perm_accs si dispo, sinon pvals
        if "perm_accs" in dd:
            perm = np.asarray(dd["perm_accs"])
            null_max = perm.max(axis=1)
            ind = max(1, int(alpha * len(null_max)))
            thr = np.sort(null_max)[-ind]
            mask = (np.asarray(dd["acc_mean"])) > thr
        elif "pvals" in dd:
            mask = np.asarray(dd["pvals"]) < alpha
        else:
            mask = np.zeros(len(acc), dtype=bool)
        im, _ = mne.viz.plot_topomap(
            acc, info, axes=ax, show=False, cmap="Reds",
            vlim=(vmin, vmax), extrapolate="head", sphere=0.11,
            mask=mask,
            mask_params=dict(marker="*", markerfacecolor="w", markeredgecolor="k",
                             markersize=8, linewidth=0),
            contours=0,
        )
        ax.set_title(f"{BAND_LABELS.get(b, b)}", fontsize=9)
        axes.append((ax, im))
    return axes, vmin, vmax


def main():
    args = parse_args()
    info = make_info(args.coord_file)

    psd_path = args.in_dir / f"fig3_psd_spectrum_{args.state}.npz"
    ttest_path = args.in_dir / f"fig3_ttest_{args.state}.npz"
    if not psd_path.exists():
        raise FileNotFoundError(psd_path)
    if not ttest_path.exists():
        raise FileNotFoundError(ttest_path)

    fig = plt.figure(figsize=(14, 8))
    # grille : 1 colonne PSD large + 5 lignes t-values + 5 lignes LDA
    gs = fig.add_gridspec(len(BANDS), 3, width_ratios=[1.4, 1, 1],
                          hspace=0.5, wspace=0.3)

    # PSD : occupe toute la colonne gauche (fusion des 5 lignes)
    ax_psd = fig.add_subplot(gs[:, 0])
    panel_psd(ax_psd, psd_path)

    # t-values : colonne du milieu (5 topomaps)
    gs_t = [gs[i, 1] for i in range(len(BANDS))]
    t_axes, tvmax = panel_tvalues(fig, gs_t, ttest_path, info, args.alpha)
    # colorbar t-values
    if t_axes:
        cax = fig.add_axes([0.63, 0.15, 0.008, 0.2])
        fig.colorbar(t_axes[-1][1], cax=cax, label="t-value")

    # LDA : colonne droite (5 topomaps), optionnel
    gs_l = [gs[i, 2] for i in range(len(BANDS))]
    if args.results is not None:
        l_axes, lvmin, lvmax = panel_lda(fig, gs_l, args.results, args.state,
                                         args.suffix, info, args.alpha)
        if l_axes:
            cax2 = fig.add_axes([0.92, 0.15, 0.008, 0.2])
            fig.colorbar(l_axes[-1][1], cax=cax2, label="Accuracy (%)")
    else:
        for i in range(len(BANDS)):
            ax = fig.add_subplot(gs_l[i])
            ax.axis("off")
            if i == 0:
                ax.set_title("LDA : --results non fourni", fontsize=9)

    # titres de colonnes
    fig.text(0.19, 0.95, "PSD", ha="center", fontsize=13, weight="bold")
    fig.text(0.55, 0.95, "Corrected T-values", ha="center", fontsize=13, weight="bold")
    fig.text(0.84, 0.95, "LDA decoding accuracy", ha="center", fontsize=13, weight="bold")
    fig.suptitle(f"Fig. 3 (Arthur chap.1) — HR vs LR, {args.state}  "
                 f"(stars: p<{args.alpha}, maxstat)", fontsize=12, y=1.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {args.out}")


if __name__ == "__main__":
    main()