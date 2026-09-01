#!/usr/bin/env python3
"""Patch de plot_barplot_psd_arthur_clean.py.

Deux choses :
  1. --pool-family : la branche sans recouvrement nomme la famille poolee
     psd_classic et non psd, le prefixe etait code en dur dans POOL_FAMILY.
  2. Texte affiche en anglais, plus de mention "facon Arthur", RFX une seule
     fois et sans parentheses, pour rester coherent avec le barplot riemannien
     deja anglicise.

Idempotent : ne fait rien si le fichier est deja patche. Sauvegarde en .bak_pf.

Usage :
    cd ~/dream-recall-alex
    python patch_psd_pool_family.py
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

P = Path("plot_barplot_psd_arthur_clean.py")

REPL: list[tuple[str, str]] = [
    # ---- 1. texte en anglais -------------------------------------------------
    (
        'LEVELS = {\n'
        '    "raw":    "non corrige (p brute, meilleure electrode)",\n'
        '    "arthur": "max-stat electrodes (Arthur, feature seule)",\n'
        '    "pooled": "max-stat pooled, psd sur 5 bandes",\n'
        '}',
        'LEVELS = {\n'
        '    "raw":    "uncorrected p, best electrode",\n'
        '    "arthur": "max-stat across the 19 electrodes of one band",\n'
        '    "pooled": "max-stat pooled over the 5 psd bands",\n'
        '}',
    ),
    (
        '    ax.set_title(\n'
        '        f"PSD bruts facon Arthur (subject/RFX), {LEVELS[level]}, p < {alpha}"\n'
        '    )',
        '    ax.set_title(\n'
        '        f"Spectral power, RFX permutation, {LEVELS[level]}, p < {alpha}"\n'
        '    )',
    ),
    (
        '        note = "- - -  seuil pooled psd   |   *  p < %.2g corrige" % alpha\n'
        '    else:\n'
        '        note = "- - -  seuil par feature   |   *  p < %.2g" % alpha',
        '        note = "- - -  pooled psd threshold   |   *  corrected p < %.2g" % alpha\n'
        '    else:\n'
        '        note = "- - -  per-feature threshold   |   *  uncorrected p < %.2g" % alpha',
    ),
    # ---- 2. --pool-family ----------------------------------------------------
    (
        'POOL_FAMILY = "psd"  # -> psd_{state}_maxstat.npz',
        'POOL_FAMILY = "psd"  # -> psd_{state}_maxstat.npz ; surchargeable par --pool-family',
    ),
    (
        '    p.add_argument("--alpha", type=float, default=0.05)',
        '    p.add_argument("--pool-family", type=str, default=POOL_FAMILY,\n'
        '                   help="Prefixe des .npz pooled : "\n'
        '                        "{pool_family}_{state}_maxstat.npz. La branche avec "\n'
        '                        "recouvrement nomme cette famille psd, la branche sans "\n'
        '                        "recouvrement la nomme psd_classic.")\n'
        '    p.add_argument("--alpha", type=float, default=0.05)',
    ),
    (
        'def load_pooled_pval(corrected_path: Path, key: str, state: str):',
        'def load_pooled_pval(corrected_path: Path, key: str, state: str,\n'
        '                     pool_family: str = POOL_FAMILY):',
    ),
    (
        'def pooled_threshold(corrected_path: Path, state: str, alpha: float):',
        'def pooled_threshold(corrected_path: Path, state: str, alpha: float,\n'
        '                     pool_family: str = POOL_FAMILY):',
    ),
    (
        'def collect(save_path: Path, corrected_path: Path, level: str, alpha: float):',
        'def collect(save_path: Path, corrected_path: Path, level: str, alpha: float,\n'
        '            pool_family: str = POOL_FAMILY):',
    ),
    (
        '                p = load_pooled_pval(corrected_path, key, state)',
        '                p = load_pooled_pval(corrected_path, key, state, pool_family)',
    ),
    (
        '            pooled_threshold(corrected_path, state, alpha) if level == "pooled" else np.nan',
        '            pooled_threshold(corrected_path, state, alpha, pool_family)\n'
        '            if level == "pooled" else np.nan',
    ),
    (
        'def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax):\n'
        '    accs, stds, sigs, bar_thr, thr_pool = collect(save_path, corrected_path, level, alpha)',
        'def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax,\n'
        '                pool_family=POOL_FAMILY):\n'
        '    accs, stds, sigs, bar_thr, thr_pool = collect(save_path, corrected_path,\n'
        '                                                  level, alpha, pool_family)',
    ),
    (
        '        make_figure(args.save_path, args.corrected_path, args.out_dir,\n'
        '                    level, args.alpha, args.ymin, args.ymax)',
        '        make_figure(args.save_path, args.corrected_path, args.out_dir,\n'
        '                    level, args.alpha, args.ymin, args.ymax, args.pool_family)',
    ),
]

# les deux occurrences du chemin pooled, remplacees ensemble
PATH_OLD = '    f = corrected_path / f"{POOL_FAMILY}_{state}_maxstat.npz"'
PATH_NEW = '    f = corrected_path / f"{pool_family}_{state}_maxstat.npz"'


def main() -> int:
    if not P.exists():
        print(f"introuvable : {P.resolve()}", file=sys.stderr)
        return 1

    src = P.read_text(encoding="utf-8")

    if "--pool-family" in src and "Spectral power, RFX permutation" in src:
        print("deja patche, rien a faire")
        return 0

    out = src
    missing = []
    for old, new in REPL:
        if old in out:
            out = out.replace(old, new, 1)
        elif new in out:
            pass  # deja applique
        else:
            missing.append(old.splitlines()[0][:70])

    n_path = out.count(PATH_OLD)
    if n_path:
        out = out.replace(PATH_OLD, PATH_NEW)
    elif PATH_NEW not in out:
        missing.append(PATH_OLD[:70])

    if missing:
        print("motifs introuvables, aucun ecriture :", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 2

    if "POOL_FAMILY}_{state}" in out:
        print("reste un POOL_FAMILY code en dur, abandon", file=sys.stderr)
        return 3

    ast.parse(out)  # garde-fou syntaxe

    shutil.copy2(P, P.with_suffix(".py.bak_pf"))
    P.write_text(out, encoding="utf-8")
    print(f"patche : {P}  (sauvegarde {P.name}.bak_pf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
