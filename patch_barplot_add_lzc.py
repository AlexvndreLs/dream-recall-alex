"""Ajoute lzc a plot_barplot_vector_clean.py : liste de features et couleur.

Deux remplacements exacts, idempotents. A lancer depuis la racine du repo,
puis verifier avec git diff avant de committer.

    cd /home/alouis/dream-recall-alex
    python patch_barplot_add_lzc.py
    git diff plot_barplot_vector_clean.py

Rien d'autre a toucher :
  - build_pvalue_summary_table.py decouvre les cles en listant results/*.npz
    et traite toute cle inconnue comme isolee (pooled == arthur), ce qui est
    exactement le statut de lzc. La ligne apparaitra sans modification.
  - classify.py route lzc vers la voie vectorielle par is_matrix_feature.
  - compute_maxstat_correction.py prend la cle en argument.
"""

from pathlib import Path

TARGET = Path("plot_barplot_vector_clean.py")

REPLACEMENTS = [
    (
        'COMPLEXITY_KEYS = ["aperiodic", "higuchi_fd", "perm_entropy", "spec_entropy"]',
        'COMPLEXITY_KEYS = ["aperiodic", "higuchi_fd", "perm_entropy", "spec_entropy", "lzc"]',
    ),
    (
        '    "spec_entropy": "#7f7f7f",\n}',
        '    "spec_entropy": "#7f7f7f",\n    "lzc": "#8c564b",\n}',
    ),
]


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} introuvable, lancer depuis la racine du repo.")

    text = TARGET.read_text()
    changed = 0
    for old, new in REPLACEMENTS:
        if new in text:
            print(f"deja applique : {old.splitlines()[0][:60]}...")
            continue
        if text.count(old) != 1:
            raise SystemExit(
                f"motif absent ou ambigu ({text.count(old)} occurrences) :\n{old}"
            )
        text = text.replace(old, new)
        changed += 1

    if changed:
        TARGET.write_text(text)
    print(f"{changed} remplacement(s) applique(s). Verifier : git diff {TARGET}")


if __name__ == "__main__":
    main()