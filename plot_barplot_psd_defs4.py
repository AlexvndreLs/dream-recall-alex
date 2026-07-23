"""Les 4 definitions de la puissance spectrale, correction max-stat 19 electrodes.

Quatre formulations comparees pour chaque bande :

  psd_{b}         P                        puissance brute Welch, V^2/Hz
  psd_osc_{b}     P / A                    ratio lineaire       (pipeline actuel)
  psd_logsub_{b}  log10(P) - log10(A)      soustraction log     (canonique specparam)
  psd_sub_{b}     P - A                    soustraction lineaire

ou A = 10 ** ap_fit_log est le fit aperiodic FOOOF reconstruit en lineaire.
Les trois dernieres partagent le meme fit ; seule l'operation de residu differe.

Ce que la figure montre
-----------------------
Les definitions ne se classent pas pareil selon la bande (diagnostic 23/07 :
r=0.39-0.45 en delta entre ratio et soustraction, 0.90-0.93 en sigma). Mettre
les quatre cote a cote rend visible quels resultats dependent du choix de
formulation et lesquels sont robustes.

Deux axes se croisent dans ces quatre definitions :
  - normalisee par le 1/f (ratio, logsub) vs non normalisee (brute, sub)
  - espace log (logsub) vs lineaire (les trois autres)
Un resultat present uniquement dans une colonne du premier axe depend de la
normalisation, ce qui est interpretable ; present uniquement dans une colonne
du second axe, cela releve de la ponderation des bins par band_power
(mean(log x) != log(mean x)).

Correction
----------
Max-stat sur les 19 electrodes, feature seule (compute_maxstat_correction.py
--mode arthur). C'est le seul niveau ou des resultats vectoriels survivent :
le pooled sur 95 tests (5 bandes) donne 0 significatif pour les QUATRE
definitions, aux quatre stades. Ce choix doit etre assume explicitement : le
max-stat 19 traite chaque bande comme une hypothese a priori independante.

Hauteur de barre = accuracy de la meilleure electrode parmi 19, convention de
plot_barplot_vector_clean.py et de la Fig. 4 d'Arthur. Error bar = acc_std
inter-bootstrap de cette electrode. Trait pointille = seuil max-stat de la
feature (quantile 1-alpha de null_max), propre a chaque barre.

Deux modes
----------
  --focus   4 combos ou quelque chose se passe (defaut) : lisible en reunion.
  --full    les 20 combos, 80 barres : dense, pour verification exhaustive.

Usage
-----
    python plot_barplot_psd_defs4.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --sub-path       /scratch/alouis/dream_features_noica_1000hz_sub \
        --logsub-path    /scratch/alouis/dream_features_noica_1000hz_logsub \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --sub-corrected-path    /scratch/alouis/dream_features_noica_1000hz_sub_corrected \
        --logsub-corrected-path /scratch/alouis/dream_features_noica_1000hz_logsub_corrected \
        --out-dir        /home/alouis/dream-recall-alex/plot_overlap \
        --alpha 0.05
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_common import (
    BANDS,
    BAND_COLORS,
    RESOLUTION,
    STATES_ORDERED,
    load_result,
    maxstat_threshold,
)

# Ordre d'affichage a l'interieur d'un combo. src designe le dossier de
# provenance, resolu par paths_for().
DEFS = [
    dict(prefix="psd_",        src="overlap", label="brute (P)",              hatch=""),
    dict(prefix="psd_osc_",    src="overlap", label="ratio (P/A)",            hatch="//"),
    dict(prefix="psd_logsub_", src="logsub",  label="log (logP - logA)",      hatch=".."),
    dict(prefix="psd_sub_",    src="sub",     label="soustraction (P - A)",   hatch="xx"),
]

# Combos retenus en mode --focus : les seuls ou au moins une definition ressort
# significative au max-stat 19 electrodes. Le reste des 20 combos est plat.
FOCUS = [("SWS", "delta"), ("SWS", "beta"), ("S2", "sigma"), ("S2", "alpha")]

WIDTH = 0.20
COMBO_STEP = 1.0
GROUP_GAP = 0.6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path",    type=Path, required=True)
    p.add_argument("--sub-path",     type=Path, required=True)
    p.add_argument("--logsub-path",  type=Path, required=True)
    p.add_argument("--corrected-path",        type=Path, required=True)
    p.add_argument("--sub-corrected-path",    type=Path, required=True)
    p.add_argument("--logsub-corrected-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--full", action="store_true", default=False,
                   help="Trace les 20 combos au lieu des 4 de FOCUS.")
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    return p.parse_args()


def paths_for(d: dict, args) -> tuple[Path, Path]:
    if d["src"] == "sub":
        return args.sub_path, args.sub_corrected_path
    if d["src"] == "logsub":
        return args.logsub_path, args.logsub_corrected_path
    return args.save_path, args.corrected_path


def arthur_entry(save: Path, corrected: Path, key: str, state: str, alpha: float):
    """acc/std/elec de la meilleure electrode + p et seuil max-stat (mode arthur).

    Retourne None si le resultat brut manque. p et thr valent None/NaN si le
    .npz maxstat_arthur est absent : la barre est tracee sans etoile ni trait
    plutot que de faire echouer la figure.
    """
    res = load_result(save, key, state)
    if res is None:
        return None

    acc = np.array(res["acc_mean"], dtype=float)
    std = np.array(res["acc_std"], dtype=float)
    best = int(np.argmax(acc))
    ch = res["ch_names"].tolist() if "ch_names" in res else list(range(len(acc)))

    f = corrected / f"{key}_{state}_maxstat_arthur.npz"
    p, thr = None, np.nan
    if f.exists():
        d = np.load(f, allow_pickle=True)
        p = float(np.array(d["pvals_corrected"], dtype=float)[best])
        thr = maxstat_threshold(d["null_max"], alpha) * 100

    return dict(acc=acc[best] * 100, std=float(std[best]) * 100,
                elec=str(ch[best]), p=p, sig=(p is not None and p < alpha),
                thr=thr)


def combos_to_plot(full: bool):
    if full:
        return [(st, b) for st in STATES_ORDERED for b in BANDS]
    return FOCUS


def collect(args):
    combos = combos_to_plot(args.full)
    cells = {}
    for st, band in combos:
        for di, d in enumerate(DEFS):
            save, corrected = paths_for(d, args)
            key = d["prefix"] + band
            e = arthur_entry(save, corrected, key, st, args.alpha)
            if e is None:
                print(f"  absent : {key}_{st}.npz")
                continue
            cells[(st, band, di)] = e
    return combos, cells


def make_figure(args):
    combos, cells = collect(args)
    if not cells:
        print("  aucune donnee, figure non generee")
        return None

    n_def = len(DEFS)
    width_in = 11 if not args.full else 22
    fig, ax = plt.subplots(figsize=(width_in, 6.0))

    offsets = [(di - (n_def - 1) / 2) * WIDTH for di in range(n_def)]

    # En mode full, un espace supplementaire entre stades pour la lisibilite.
    xs, labels, group_marks = [], [], []
    x = 0.0
    prev_state = None
    for st, band in combos:
        if prev_state is not None and st != prev_state:
            x += GROUP_GAP
        xs.append(x)
        labels.append(band if args.full else f"{band}\n{st}")
        group_marks.append(st)
        prev_state = st
        x += COMBO_STEP

    n_sig = 0
    for ci, (st, band) in enumerate(combos):
        xc = xs[ci]
        for di, d in enumerate(DEFS):
            c = cells.get((st, band, di))
            if c is None:
                continue
            xb = xc + offsets[di]
            ax.bar(xb, c["acc"], WIDTH,
                   color=BAND_COLORS[band], hatch=d["hatch"],
                   edgecolor="white", linewidth=0.7,
                   yerr=c["std"], capsize=1.5, error_kw=dict(lw=0.8))
            if c["sig"]:
                n_sig += 1
                ax.text(xb, c["acc"] + c["std"] + 0.35, "*", ha="center",
                        va="bottom", fontsize=14, fontweight="bold")
            if not np.isnan(c["thr"]):
                ax.plot([xb - WIDTH / 2, xb + WIDTH / 2], [c["thr"]] * 2,
                        "k--", lw=1.1, zorder=3)
            # Nom de l'electrode retenue, sous la barre : la topographie change
            # d'une definition a l'autre (P3 en log, O2 en ratio pour beta/SWS).
            if not args.full:
                ax.text(xb, ax.get_ylim()[0], c["elec"], ha="center", va="bottom",
                        fontsize=6, rotation=90, color="0.25")

    vals = [c["acc"] for c in cells.values()]
    lo = args.ymin if args.ymin is not None else min(48, min(vals) - 3)
    hi = args.ymax if args.ymax is not None else max(vals) + 5
    ax.set_ylim(lo, hi)

    # Les etiquettes d'electrode ont ete posees avant le set_ylim final ; on les
    # repositionne en coordonnees data maintenant que l'axe est fige.
    if not args.full:
        for t in ax.texts:
            if t.get_rotation() == 90:
                t.set_y(lo + 0.15)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    if args.full:
        seen = set()
        for ci, st in enumerate(group_marks):
            if st in seen:
                continue
            seen.add(st)
            idx = [i for i, s in enumerate(group_marks) if s == st]
            ax.text((xs[idx[0]] + xs[idx[-1]]) / 2, lo - (hi - lo) * 0.09,
                    st, ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_ylabel("Decoding accuracy (%)")
    mode = "20 combos" if args.full else "combos significatifs"
    ax.set_title(f"Quatre definitions de la puissance spectrale ({mode}), "
                 f"max-stat 19 electrodes, p < {args.alpha}")
    ax.axhline(50, color="gray", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(handles=[Patch(facecolor="0.75", hatch=d["hatch"],
                             edgecolor="white", label=d["label"]) for d in DEFS],
              frameon=False, fontsize=9, ncol=2, loc="upper right")

    ax.text(0.995, 0.02,
            "- - -  seuil max-stat par feature   |   *  p < %.2g" % args.alpha,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.3")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if args.full else "focus"
    out = args.out_dir / f"barplot_psd_defs4_arthur_{suffix}_p{args.alpha}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=RESOLUTION, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  {n_sig} barres significatives -> {out}")
    return cells


def print_table(cells, combos):
    print("\n=== max-stat 19 electrodes : acc%% (electrode) p ===")
    hdr = f"{'state':6s} {'band':7s}"
    for d in DEFS:
        hdr += f" {d['label'][:14]:>22s}"
    print(hdr)
    print("-" * len(hdr))
    for st, band in combos:
        row = f"{st:6s} {band:7s}"
        for di in range(len(DEFS)):
            c = cells.get((st, band, di))
            if c is None:
                row += f" {'--':>22s}"
                continue
            star = "*" if c["sig"] else " "
            pv = f"{c['p']:.3f}" if c["p"] is not None else "  na "
            row += f" {c['acc']:6.2f} {c['elec']:>4s} {pv:>6s}{star}"
        print(row)


def main() -> None:
    args = parse_args()
    print(f"=== 4 definitions, max-stat arthur, p < {args.alpha} ===")
    cells = make_figure(args)
    if cells:
        print_table(cells, combos_to_plot(args.full))


if __name__ == "__main__":
    main()
