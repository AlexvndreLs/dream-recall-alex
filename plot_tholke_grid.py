"""Grilles topographiques au format de la figure 3 de Tholke et al. 2025.

Reference
---------
Tholke, Arcand-Lavigne, Lajnef, Frenette, Carrier, Jerbi (2025). Caffeine
induces age-dependent increases in brain complexity and criticality during
sleep. Communications Biology 8:685.

Trois ecarts assumes avec leur figure, a rappeler dans la legende du rapport :


1. Pas de colonne SVM. Leur figure en a une, pas nous : la these d'Arthur
   (§1.2.7 Classifier Selection) indique que SVM, LDA et KNN ont ete evalues
   mais que seule la LDA a ete retenue. Aucun resultat SVM n'existe dans le
   pipeline. Deux colonnes par etat au lieu de trois.

2. Quatre etats de vigilance au lieu de deux : S2, SWS, NREM, REM.

3. Contraste inter-groupes et non apparie. Leur t-test compare deux nuits du
   meme sujet, le notre oppose 18 hauts rappeleurs a 18 bas rappeleurs.
   Structurellement moins puissant, les amplitudes de t ne se comparent pas
   terme a terme.

Conventions reprises telles quelles
-----------------------------------
Code couleur signe pour les t : bleu valeur plus basse chez les hauts
rappeleurs, rouge plus haute. Vert sequentiel pour l'accuracy, depart au
niveau de la chance. Marquage a deux seuils, gris p < 0.05, blanc p < 0.01,
sur les p CORRIGEES dans les deux colonnes. C'est bien ce que fait leur code :
mne.stats.permutation_t_test corrige par max-stat sur la dimension des canaux,
et leur notebook de figures prend le maximum sur les electrodes des
distributions nulles d'accuracy avant de seuiller a 5 % et 1 %. La portee de
correction est donc l'electrode a l'interieur d'un couple mesure x etat, jamais
un pool entre mesures. C'est aussi la portee du mode 'arthur' de
compute_maxstat_correction.py et celle de ttest_vector_rfx.py.

Entrees attendues
-----------------
    {root}/results/{key}_{state}.npz                       -> acc_mean
    {root}_corrected/{key}_{state}_maxstat_arthur.npz      -> pvals_corrected
    {root}_ttest/{key}_{state}_ttest_rfx.npz               -> tvals, pvals_corrected

Une cellule dont un fichier manque est tracee en gris avec la mention n/a,
plutot que d'interrompre la figure. Utile tant que la LDA de l'offset tourne.

Usage
-----
    # figure apериodique : exposant et offset, quatre etats
    python plot_tholke_grid.py --figure aperiodic \
        --out-dir final_plotted_figures

    # figures oscillatoires : une par etat, quatre definitions x cinq bandes
    python plot_tholke_grid.py --figure oscillatory \
        --out-dir final_plotted_figures

    # avec le montage exact d'Arthur
    python plot_tholke_grid.py --figure aperiodic \
        --coord-file /home/alouis/dream-recall-alex/coord_cart_new.txt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mne.channels.layout import _find_topomap_coords

# CH_NAMES vient de config_v3, jamais recopie ici. Les 19 canaux EEG ne sont pas
# le 10-20 classique : il y a FC1/FC2/CP1/CP2 et pas de F7/F8/T5/T6, et l'ordre
# n'est pas alphabetique ni topographique. Recopier cette liste a la main revient
# a placer chaque valeur sur la mauvaise electrode, sans que la figure ait l'air
# fausse pour autant.
try:
    from config_v3 import CH_NAMES as _CH_ALL, N_EEG
    CH_NAMES = list(_CH_ALL[:N_EEG])
except ImportError as exc:  # execution hors du repo
    raise SystemExit(
        "config_v3.py introuvable. Lancer ce script depuis le repo "
        "dream-recall-alex, la liste des electrodes et leur ordre en "
        "dependent."
    ) from exc

OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

STATES = ["S2", "SWS", "NREM", "REM"]
BANDS = ["delta", "theta", "alpha", "sigma", "beta"]

# Les quatre definitions de puissance oscillatoire vivent dans trois racines
# differentes. sub et logsub portent un nom sans "_overlap" mais ont bien ete
# extraites avec OVERLAP = 500 : leurs batchs pointent --cov-source vers la
# branche overlap, et feat_extract_sub.py importe compute_psd_spectrum du
# module principal, qui lit OVERLAP dans la config au moment de l'import.
DEFINITIONS = {
    "Raw power":        ("dream_features_noica_1000hz_overlap", "psd_{band}"),
    "Ratio":            ("dream_features_noica_1000hz_overlap", "psd_osc_{band}"),
    "Subtracted":       ("dream_features_noica_1000hz_sub", "psd_sub_{band}"),
    "Log-subtracted":   ("dream_features_noica_1000hz_logsub", "psd_logsub_{band}"),
}

APERIODIC_ROWS = [
    ("Aperiodic exponent", "dream_features_noica_1000hz_overlap", "aperiodic"),
    ("Aperiodic offset",   "dream_features_noica_1000hz_overlap", "aperiodic_offset"),
]

# Vert sequentiel proche du leur : blanc a la chance, vert fonce au maximum.
GREEN = LinearSegmentedColormap.from_list(
    "tholke_green", ["#ffffff", "#d4ead6", "#8fc99a", "#3f9455", "#12592a"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--figure", choices=["aperiodic", "oscillatory"], required=True)
    p.add_argument("--scratch", type=Path, default=Path("/scratch/alouis"))
    p.add_argument("--out-dir", type=Path, default=Path("final_plotted_figures"))
    p.add_argument("--coord-file", type=Path, default=None,
                   help="coord_cart_new.txt d'Arthur. Sans lui, montage "
                        "standard_1020 : positions tres proches, bords "
                        "legerement differents.")
    p.add_argument("--sphere", type=float, default=0.11)
    p.add_argument("--tmax", type=float, default=None,
                   help="Borne symetrique de l'echelle des t. Par defaut, "
                        "calculee sur la figure et arrondie au demi superieur. "
                        "Ne pas reprendre le +-5 de Tholke : nos |t| plafonnent "
                        "bien plus bas, la figure serait presque blanche.")
    p.add_argument("--accmax", type=float, default=None,
                   help="Borne haute de l'echelle d'accuracy, en pourcentage.")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def make_info(coord_file: Path | None) -> mne.Info:
    """Info MNE des 19 electrodes. Meme logique que
    plot_topomap_psd_arthur_exact.py, y compris la permutation de repere."""
    if coord_file is not None:
        coords = np.loadtxt(coord_file)
        if coords.shape != (N_EEG, 3):
            raise ValueError(f"{coord_file} : attendu ({N_EEG}, 3), "
                             f"lu {coords.shape}.")
        xa, ya, za = coords[:, 0], coords[:, 1], coords[:, 2]
        coords_mne = np.column_stack([-ya, xa, za])
        pos = {ch: coords_mne[i] * 0.095 for i, ch in enumerate(CH_NAMES)}
        info = mne.create_info(list(CH_NAMES), sfreq=1.0, ch_types="eeg")
        info.set_montage(mne.channels.make_dig_montage(ch_pos=pos,
                                                       coord_frame="head"))
        return info

    names = [OLD_TO_NEW.get(ch, ch) for ch in CH_NAMES]
    info = mne.create_info(names, sfreq=1.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=False)
    return info


def load_cell(scratch: Path, branch: str, key: str, state: str) -> dict:
    """Charge les quatre vecteurs d'une cellule. Valeur None si absente."""
    out = dict(tvals=None, p_t=None, acc=None, p_lda=None)

    f = scratch / f"{branch}_ttest" / f"{key}_{state}_ttest_rfx.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            out["tvals"] = z["tvals"].copy()
            out["p_t"] = z["pvals_corrected"].copy()

    f = scratch / branch / "results" / f"{key}_{state}.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            out["acc"] = z["acc_mean"].copy() * 100.0

    f = scratch / f"{branch}_corrected" / f"{key}_{state}_maxstat_arthur.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            out["p_lda"] = z["pvals_corrected"].copy()

    return out


def draw_topo(ax, values, pvals, info, pos, cmap, vlim, sphere):
    """Une topographie plus son marquage a deux seuils.

    Le marquage n'utilise pas le parametre mask de plot_topomap, qui ne sait
    porter qu'un seul style : on desactive les capteurs et on trace les trois
    niveaux a la main. Petit point noir partout, rond gris p < 0.05, rond blanc
    p < 0.01, comme chez Tholke.
    """
    if values is None:
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color="0.5")
        ax.axis("off")
        return None

    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap=cmap, vlim=vlim,
        sensors=False, contours=0, sphere=sphere, res=128,
    )
    ax.scatter(pos[:, 0], pos[:, 1], s=1.6, c="k", zorder=5)

    if pvals is not None:
        weak = (pvals < 0.05) & (pvals >= 0.01)
        strong = pvals < 0.01
        if weak.any():
            ax.scatter(pos[weak, 0], pos[weak, 1], s=32, facecolor="0.62",
                       edgecolor="k", linewidth=0.5, zorder=6)
        if strong.any():
            ax.scatter(pos[strong, 0], pos[strong, 1], s=32, facecolor="white",
                       edgecolor="k", linewidth=0.5, zorder=6)
    return im


def build_grid(cells, row_labels, col_groups, info, pos, args, title, out_png):
    """Trace une grille lignes x (groupes de colonnes x [T, LDA]).

    cells[(r, c)] est le dict renvoye par load_cell.
    col_groups est la liste des libelles de groupe, etats ou bandes.
    """
    n_rows = len(row_labels)
    n_groups = len(col_groups)
    n_cols = 2 * n_groups

    all_t = [c["tvals"] for c in cells.values() if c["tvals"] is not None]
    all_a = [c["acc"] for c in cells.values() if c["acc"] is not None]
    tmax = args.tmax if args.tmax else (
        np.ceil(np.abs(np.concatenate(all_t)).max() * 2) / 2 if all_t else 3.0)
    amax = args.accmax if args.accmax else (
        np.ceil(np.concatenate(all_a).max()) if all_a else 65.0)
    tlim, alim = (-tmax, tmax), (50.0, amax)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.55 * n_cols + 1.6, 1.65 * n_rows + 2.0),
        gridspec_kw=dict(wspace=0.05, hspace=0.10,
                         left=0.13, right=0.99, top=0.80, bottom=0.22))
    axes = np.atleast_2d(axes)

    im_t = im_a = None
    for r in range(n_rows):
        for g in range(n_groups):
            cell = cells[(r, g)]
            ax = axes[r, 2 * g]
            res = draw_topo(ax, cell["tvals"], cell["p_t"], info, pos,
                            "RdBu_r", tlim, args.sphere)
            im_t = res if res is not None else im_t

            ax = axes[r, 2 * g + 1]
            res = draw_topo(ax, cell["acc"], cell["p_lda"], info, pos,
                            GREEN, alim, args.sphere)
            im_a = res if res is not None else im_a

    for g, label in enumerate(col_groups):
        for k, sub in enumerate(["T-values", "LDA"]):
            axes[0, 2 * g + k].set_title(sub, fontsize=9, pad=4)
        x0 = axes[0, 2 * g].get_position().x0
        x1 = axes[0, 2 * g + 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.885, label, ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

    for r, label in enumerate(row_labels):
        y = (axes[r, 0].get_position().y0 + axes[r, 0].get_position().y1) / 2
        fig.text(0.115, y, label, ha="right", va="center", fontsize=10)

    cax_t = fig.add_axes([0.17, 0.115, 0.28, 0.020])
    cb_t = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*tlim), cmap="RdBu_r"),
        cax=cax_t, orientation="horizontal")
    cb_t.set_label("t (high recallers minus low recallers)", fontsize=8,
                   labelpad=2)
    cb_t.set_ticks([-tmax, 0, tmax])
    cb_t.ax.set_xticklabels([f"{-tmax:g}", "0", f"{tmax:g}"])
    cb_t.ax.tick_params(labelsize=8, pad=1)

    cax_a = fig.add_axes([0.60, 0.115, 0.28, 0.020])
    cb_a = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*alim), cmap=GREEN),
        cax=cax_a, orientation="horizontal")
    cb_a.set_label("LDA decoding accuracy (%)", fontsize=8, labelpad=2)
    ticks = np.linspace(alim[0], alim[1], 4)
    cb_a.set_ticks(ticks)
    cb_a.ax.set_xticklabels([f"{t:.0f}" for t in ticks])
    cb_a.ax.tick_params(labelsize=8, pad=1)

    fig.text(0.5, 0.955, title, ha="center", va="center", fontsize=13,
             fontweight="bold")
    fig.text(0.5, 0.035,
             "Gray dots: p < 0.05.   White dots: p < 0.01.   "
             "Max-statistic correction over the 19 electrodes, "
             "within each measure and stage.",
             ha="center", va="center", fontsize=8, color="0.3")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"  -> {out_png}   (t: +-{tmax:.1f}, acc: {alim[0]:.0f} a {alim[1]:.0f} %)")
    return tmax, amax


def main() -> None:
    args = parse_args()
    info = make_info(args.coord_file)
    pos = _find_topomap_coords(info, picks=np.arange(N_EEG), sphere=args.sphere)

    if args.figure == "aperiodic":
        cells = {}
        for r, (_, branch, key) in enumerate(APERIODIC_ROWS):
            for g, state in enumerate(STATES):
                cells[(r, g)] = load_cell(args.scratch, branch, key, state)
        out = args.out_dir / "fig13_aperiodic_topo.png"
        print("figure aperiodique")
        build_grid(cells, [lab for lab, _, _ in APERIODIC_ROWS], STATES,
                   info, pos, args,
                   "Aperiodic background: high vs low dream recallers", out)
        return

    # Les quatre figures oscillatoires doivent partager la meme echelle, sinon
    # une topographie plus rouge dans un etat que dans un autre ne voudrait
    # rien dire. On charge tout d'abord, on fixe les bornes, puis on trace.
    per_state = {}
    for state in STATES:
        cells = {}
        for r, (_, (branch, pat)) in enumerate(DEFINITIONS.items()):
            for g, band in enumerate(BANDS):
                cells[(r, g)] = load_cell(args.scratch, branch,
                                          pat.format(band=band), state)
        per_state[state] = cells

    if args.tmax is None:
        allt = [c["tvals"] for cs in per_state.values() for c in cs.values()
                if c["tvals"] is not None]
        if allt:
            args.tmax = float(np.ceil(np.abs(np.concatenate(allt)).max() * 2) / 2)
    if args.accmax is None:
        alla = [c["acc"] for cs in per_state.values() for c in cs.values()
                if c["acc"] is not None]
        if alla:
            args.accmax = float(np.ceil(np.concatenate(alla).max()))

    for state in STATES:
        out = args.out_dir / f"fig14_oscillatory_topo_{state}.png"
        print(f"figure oscillatoire, {state}")
        build_grid(per_state[state], list(DEFINITIONS),
                   [b.capitalize() for b in BANDS],
                   info, pos, args,
                   f"Oscillatory power in {state}: "
                   f"high vs low dream recallers", out)


if __name__ == "__main__":
    main()