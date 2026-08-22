
#!/usr/bin/env python3
"""Traduit en anglais le texte RENDU sur les figures du rapport.

Perimetre volontairement restreint : uniquement les chaines qui apparaissent
sur l'image finale (titres, labels de legende, annotations). Les commentaires,
docstrings, messages d'aide argparse et sorties stdout restent en francais.

Idempotent : relancer le script ne fait rien si les patchs sont deja appliques.
Chaque remplacement est verifie (presence unique) avant ecriture, le script
s'arrete a la premiere anomalie sans rien modifier du fichier concerne.

Usage :
    cd ~/dream-recall-alex
    python patch_figure_labels_en.py          # applique
    python patch_figure_labels_en.py --dry    # montre sans ecrire
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Table des remplacements : fichier -> liste de (ancien, nouveau).
# Les chaines doivent correspondre EXACTEMENT au contenu du fichier, espaces
# et indentation compris.
# ---------------------------------------------------------------------------

PATCHES: dict[str, list[tuple[str, str]]] = {

    # ---------------------------------------------------- fig04, barplot FFX
    "plot_barplot_riemann_ffx_fixed.py": [
        (
            'GRAPH_TITLE = "Riemannian classifications, FFX '
            '(X figé, réplication Arthur)"',
            'GRAPH_TITLE = "Riemannian classification, FFX '
            '(fixed X, Arthur replication)"',
        ),
        (
            '    CORR_TITLE = {"none": "", "state": ", max-stat par stade",\n'
            '                  "global": ", max-stat global (24 combos)"}',
            '    CORR_TITLE = {"none": "", "state": ", per-stage max-stat",\n'
            '                  "global": ", global max-stat (24 combos)"}',
        ),
    ],

    # ------------------------------------ fig04 variante, barplot epoch brut
    "plot_barplot_riemann_epoch_corr.py": [
        (
            '    CORR_TITLE = {"none": "", "state": ", max-stat par stade",\n'
            '                  "global": ", max-stat global (24 combos)"}',
            '    CORR_TITLE = {"none": "", "state": ", per-stage max-stat",\n'
            '                  "global": ", global max-stat (24 combos)"}',
        ),
    ],

    # --------------------------------------------- fig05, topomap PSD Arthur
    "plot_topomap_psd_arthur.py": [
        (
            'f"{args.feature_family}, accuracy par électrode "',
            'f"{args.feature_family}, accuracy per electrode "',
        ),
    ],

    # ------------------------------- fig05 variante, topomap propre (RFX)
    "plot_topomap_clean.py": [
        (
            'f"{args.family}, accuracy par électrode, permutation sujet '
            '(RFX)\\n"\n'
            '        f"* : p < {args.alpha} après max-stat pooled sur '
            '{len(BANDS) * N_EEG} tests",',
            'f"{args.family}, accuracy per electrode, subject-level '
            'permutation (RFX)\\n"\n'
            '        f"* : p < {args.alpha} after pooled max-stat over '
            '{len(BANDS) * N_EEG} tests",',
        ),
    ],

    # ------------------------------------ fig09, quatre definitions de la PSD
    "plot_barplot_psd_defs4.py": [
        (
            '    dict(prefix="psd_",        src="overlap", '
            'label="brute (P)",              hatch=""),',
            '    dict(prefix="psd_",        src="overlap", '
            'label="raw (P)",                hatch=""),',
        ),
        (
            '    dict(prefix="psd_sub_",    src="sub",     '
            'label="soustraction (P - A)",   hatch="xx"),',
            '    dict(prefix="psd_sub_",    src="sub",     '
            'label="subtraction (P - A)",    hatch="xx"),',
        ),
        (
            '    mode = "20 combos" if args.full else "combos significatifs"\n'
            '    ax.set_title(f"Quatre definitions de la puissance spectrale '
            '({mode}), "\n'
            '                 f"max-stat 19 electrodes, p < {args.alpha}")',
            '    mode = "20 combos" if args.full else "significant combos"\n'
            '    ax.set_title(f"Four definitions of spectral power ({mode}), "\n'
            '                 f"max-stat over 19 electrodes, '
            'p < {args.alpha}")',
        ),
        (
            '            "- - -  seuil max-stat par feature   |   *  '
            'p < %.2g" % args.alpha,',
            '            "- - -  per-feature max-stat threshold   |   *  '
            'p < %.2g" % args.alpha,',
        ),
    ],

    # ------------------------------------------------- fig08, grille EFS ROI
    "plot_fig5_arthur_grid.py": [
        (
            'axes[r, 0].annotate(f"{stage}\\n(non calcule)", xy=(0.5, 0.5),',
            'axes[r, 0].annotate(f"{stage}\\n(not computed)", xy=(0.5, 0.5),',
        ),
        (
            '        fig.suptitle("Fig. 5 (Arthur chap.1) - EFS holdout par '
            'ROI et stade  "\n'
            '                     f"(pie: selection rate combinaisons, seuil '
            '{int(args.sr_threshold*100)}%)",',
            '        fig.suptitle("Fig. 5 (Arthur chap.1) - EFS holdout by '
            'ROI and stage  "\n'
            '                     f"(pie: combination selection rate, '
            'threshold {int(args.sr_threshold*100)}%)",',
        ),
    ],
}


def apply_file(path: Path, repls: list[tuple[str, str]], dry: bool) -> int:
    """Applique les remplacements d'un fichier. Retourne le nombre applique."""
    if not path.exists():
        print(f"[SKIP] {path.name} : fichier absent")
        return 0

    text = path.read_text(encoding="utf-8")
    original = text
    applied = 0

    for old, new in repls:
        n_old = text.count(old)
        n_new = text.count(new)

        if n_old == 0 and n_new >= 1:
            print(f"  [deja fait] {path.name} : {new.strip()[:60]}...")
            continue
        if n_old == 0:
            print(f"  [ECHEC] {path.name} : motif introuvable\n"
                  f"          ---> {old.strip()[:90]}")
            return -1
        if n_old > 1:
            print(f"  [ECHEC] {path.name} : motif present {n_old} fois, "
                  f"remplacement ambigu\n          ---> {old.strip()[:90]}")
            return -1

        text = text.replace(old, new)
        applied += 1
        print(f"  [ok] {path.name} : {new.strip()[:70]}")

    if applied and not dry and text != original:
        backup = path.with_suffix(path.suffix + ".bak_fr")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")

    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="Affiche les remplacements sans ecrire.")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="Racine du repo (defaut : repertoire courant).")
    args = ap.parse_args()

    total = 0
    failed: list[str] = []

    for fname, repls in PATCHES.items():
        print(f"\n=== {fname}")
        n = apply_file(args.root / fname, repls, args.dry)
        if n < 0:
            failed.append(fname)
        else:
            total += n

    print(f"\n{'=' * 60}")
    print(f"{total} remplacement(s) applique(s)"
          f"{' (dry run, rien ecrit)' if args.dry else ''}")

    if failed:
        print(f"ECHEC sur : {', '.join(failed)}")
        print("Ces fichiers n'ont PAS ete modifies. Verifie qu'ils "
              "correspondent bien a la version main du repo.")
        sys.exit(1)

    if not args.dry and total:
        print("Sauvegardes ecrites en *.py.bak_fr")
        print("Verifie ensuite : git diff && python3 -m py_compile "
              + " ".join(PATCHES))


if __name__ == "__main__":
    main()
