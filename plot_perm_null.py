"""Distributions nulles de permutation : histogramme + accuracy réelle.

C'est la figure qui valide statistiquement les résultats. Le barplot dit "68%",
mais 68% n'est interprétable que rapporté à ce que le pipeline produit en
l'absence de tout signal : la loi nulle. Elle est obtenue en permutant les
labels HR/LR AU NIVEAU SUJET (schéma RFX) et en refaisant toute la
classification, 1000 fois.

La p-value affichée est celle stockée dans le .npz (non corrigée). Le seuil
corrigé (max-stat pooled sur la famille) est tracé en plus quand
--corrected-path est fourni : c'est la barre à franchir une fois les
comparaisons multiples prises en compte, et elle est nettement plus haute.

Deux modes :
  --mode grid     planche de tous les combos d'une famille (défaut)
  --mode zoom     une figure large par combo listé dans --features

Usage :
    # planche des 24 combos matriciels
    python plot_perm_null.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /scratch/alouis/dream-recall-alex/plot_overlap \
        --family matrix

    # zoom sur les gagnants et un contre-exemple
    python plot_perm_null.py \
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \
        --out-dir        /scratch/alouis/dream-recall-alex/plot_overlap \
        --mode zoom --features cosp_sigma/S2 cosp_delta/SWS cov/REM
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    FAMILY_KEYS,
    RESOLUTION,
    STATES_ORDERED,
    band_label,
    is_matrix_key,
    key_color,
    load_maxstat,
    load_null_max,
    load_result,
    maxstat_threshold,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--corrected-path", type=Path, default=None,
                   help="Dossier des .npz maxstat. Sans lui, pas de seuil corrigé tracé.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--family", default="matrix", choices=list(FAMILY_KEYS),
                   help="Famille à tracer en mode grid.")
    p.add_argument("--mode", default="grid", choices=["grid", "zoom"])
    p.add_argument("--features", nargs="+", default=None,
                   help="Mode zoom : liste 'feature/state' (ex: cosp_sigma/S2).")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def real_and_null(d, key: str):
    """Extrait (accuracy réelle, distribution nulle) d'un .npz, en %.

    Pour une feature vectorielle, acc_mean est un vecteur de 19 électrodes et
    perm_accs est (n_perm, 19) : on prend la MEILLEURE électrode et sa propre
    loi nulle. Tracer les 19 superposées serait illisible, et la moyenne sur
    électrodes n'a pas de sens statistique ici.
    """
    if is_matrix_key(key):
        return float(d["acc_mean"]) * 100, np.asarray(d["perm_accs"]) * 100, None

    acc = np.asarray(d["acc_mean"])
    best = int(acc.argmax())
    ch = str(d["ch_names"][best]) if "ch_names" in d else f"élec {best}"
    return float(acc[best]) * 100, np.asarray(d["perm_accs"])[:, best] * 100, ch


def draw_one(ax, d, key: str, state: str, threshold: float | None,
             alpha: float, compact: bool) -> bool:
    """Trace un histogramme de nulle + ligne réelle. Retourne True si significatif."""
    acc, null, ch = real_and_null(d, key)

    ax.hist(null, bins=40, color="0.75", edgecolor="none")
    ax.axvline(acc, color=key_color(key), lw=2, zorder=3)

    sig = threshold is not None and acc > threshold
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1, zorder=2)

    # p-value non corrigée telle que stockée par classify.py
    pval = float(d["pval"]) if "pval" in d.files else None

    title = f"{band_label(key)}, {state}"
    if ch is not None:
        title += f" ({ch})"
    ax.set_title(title, fontsize=9 if compact else 12)

    txt = f"acc = {acc:.1f}%"
    if pval is not None:
        txt += f"\np = {pval:.3f}"
    if sig:
        txt += "  *"
    ax.text(0.97, 0.93, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=7 if compact else 10,
            bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.8", alpha=0.85))

    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    return sig


def mode_grid(args) -> None:
    keys = FAMILY_KEYS[args.family]
    print(f"=== nulles de permutation, famille {args.family} (subject) ===")

    fig, axes = plt.subplots(len(keys), len(STATES_ORDERED),
                             figsize=(3.0 * len(STATES_ORDERED), 1.9 * len(keys)),
                             sharex="col")
    axes = np.atleast_2d(axes)

    n_sig = 0
    for c, state in enumerate(STATES_ORDERED):
        null_max = (load_null_max(args.corrected_path, args.family, state)
                    if args.corrected_path else None)
        thr = maxstat_threshold(null_max, args.alpha) * 100 if null_max is not None else None

        for r, key in enumerate(keys):
            ax = axes[r, c]
            d = load_result(args.save_path, key, state)
            if d is None:
                print(f"  absent : {key}_{state}.npz")
                ax.axis("off")
                continue
            if draw_one(ax, d, key, state, thr, args.alpha, compact=True):
                n_sig += 1
            if r == len(keys) - 1:
                ax.set_xlabel("Accuracy (%)", fontsize=8)

    fig.suptitle(
        f"Distributions nulles par permutation sujet (RFX), famille {args.family}\n"
        f"gris : 1000 permutations   |   trait plein : accuracy réelle   |   "
        f"- - - : seuil max-stat pooled p < {args.alpha}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"perm_null_{args.family}_subject_p{args.alpha}.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"  {n_sig} combos significatifs (max-stat pooled)")
    print(f"Écrit : {out}")


def mode_zoom(args) -> None:
    if not args.features:
        raise SystemExit("--mode zoom exige --features feature/state [...]")

    pairs = []
    for tok in args.features:
        if "/" not in tok:
            raise SystemExit(f"--features : format attendu feature/state, reçu '{tok}'")
        feat, state = tok.split("/", 1)
        pairs.append((feat.strip(), state.strip()))

    print(f"=== nulles de permutation, zoom sur {len(pairs)} combos (subject) ===")

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 3.6))
    axes = np.atleast_1d(axes)

    for ax, (key, state) in zip(axes, pairs):
        d = load_result(args.save_path, key, state)
        if d is None:
            print(f"  absent : {key}_{state}.npz")
            ax.axis("off")
            continue

        # Le seuil dépend de la famille à laquelle appartient la feature : on la
        # retrouve plutôt que de la demander en argument.
        thr = None
        if args.corrected_path:
            for fam, fam_keys in FAMILY_KEYS.items():
                if key in fam_keys:
                    nm = load_null_max(args.corrected_path, fam, state)
                    if nm is not None:
                        thr = maxstat_threshold(nm, args.alpha) * 100
                    break

        draw_one(ax, d, key, state, thr, args.alpha, compact=False)
        ax.set_xlabel("Accuracy (%)")

    fig.suptitle(
        f"Distributions nulles par permutation sujet (RFX)   |   "
        f"- - - : seuil max-stat pooled p < {args.alpha}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{k}-{s}" for k, s in pairs)
    out = args.out_dir / f"perm_null_zoom_{tag}_p{args.alpha}.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


def main() -> None:
    args = parse_args()
    if args.mode == "grid":
        mode_grid(args)
    else:
        mode_zoom(args)


if __name__ == "__main__":
    main()