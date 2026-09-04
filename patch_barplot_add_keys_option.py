"""Ajoute --keys, --levels, --out-name et --title a plot_barplot_vector_clean.py.

Motivation
----------
La figure fig:complexite du rapport ne montre QUE les mesures de complexite,
alors que le script trace les 9 features vectorielles modernisees (5 psd_osc
plus 4 complexites). Plutot que d'ecrire un second script qui dupliquerait la
logique des seuils par barre, des etoiles et des couleurs, on ouvre le script
existant sur quatre points :

  --keys      restreint les features tracees. Sans lui, comportement inchange.
  --levels    restreint les niveaux de correction produits. La figure du
              rapport n'a besoin que de arthur, inutile d'en generer trois.
  --out-name  nom de fichier exact, pour deposer directement
              fig15_complexite_barplot.png sans renommage manuel.
  --title     titre de la figure. Le titre par defaut parle des "features
              vectorielles modernisees", ce qui est faux pour une figure
              restreinte a la complexite.

Aucun defaut n'est modifie : sans ces options le script produit exactement les
memes trois figures qu'avant. Verifie par git diff.

Le mecanisme des seuils est inchange et reste celui que decrit la legende du
rapport : en niveau arthur, le trait pointille de chaque barre est le quantile
(1-alpha) de null_max, c'est-a-dire du maximum sur les 19 electrodes de la
feature seule, lu dans {key}_{state}_maxstat_arthur.npz. C'est bien la
statistique du maximum sur les 19 electrodes, donc le meme niveau de
correction que l'etude de reference.

Usage
-----
    cd /home/alouis/dream-recall-alex
    python patch_barplot_add_keys_option.py
    git diff plot_barplot_vector_clean.py

Puis, pour la figure du rapport :

    python plot_barplot_vector_clean.py \\
        --save-path      /scratch/alouis/dream_features_noica_1000hz_overlap \\
        --corrected-path /scratch/alouis/dream_features_noica_1000hz_overlap_corrected \\
        --out-dir        /home/alouis/dream-recall-alex/final_plotted_figures \\
        --keys   perm_entropy higuchi_fd spec_entropy lzc \\
        --levels arthur \\
        --out-name fig15_complexite_barplot.png \\
        --title "Decoding accuracy of the four complexity measures (subject-level permutation)" \\
        --alpha 0.05

Note sur --keys : les quatre mesures de la section 3 sont perm_entropy,
higuchi_fd, spec_entropy et lzc. aperiodic releve de la section 2 et n'est
donc pas dans cette liste, bien qu'il figure dans COMPLEXITY_KEYS du script.
Ajouter aperiodic a --keys si la figure doit le montrer en reference.
"""

from pathlib import Path

TARGET = Path("plot_barplot_vector_clean.py")

REPLACEMENTS = [
    # 1. keys_for_level accepte une liste explicite
    (
        '''def keys_for_level(level: str):''',
        '''def keys_for_level(level: str, override=None):''',
    ),
    (
        '''    return list(PSD_OSC_KEYS) if level == "pooled" else list(VECTOR_KEYS)''',
        '''    if override:
        return list(override)
    return list(PSD_OSC_KEYS) if level == "pooled" else list(VECTOR_KEYS)''',
    ),
    # 2. nouveaux arguments CLI
    (
        '''    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    return p.parse_args()''',
        '''    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--keys", nargs="+", default=None,
                   help="Restreint les features tracees. Defaut : les 9 "
                        "features vectorielles (5 psd_osc + complexites).")
    p.add_argument("--levels", nargs="+", default=["raw", "arthur", "pooled"],
                   choices=["raw", "arthur", "pooled"],
                   help="Niveaux de correction a produire.")
    p.add_argument("--out-name", default=None,
                   help="Nom de fichier exact. Defaut : "
                        "barplot_vector_{level}_p{alpha}.png")
    p.add_argument("--title", default=None,
                   help="Titre de la figure. Defaut : titre generique sur les "
                        "features vectorielles modernisees.")
    return p.parse_args()''',
    ),
    # 3. collect propage la restriction
    (
        '''def collect(save_path: Path, corrected_path: Path, level: str, alpha: float):''',
        '''def collect(save_path: Path, corrected_path: Path, level: str, alpha: float,
            keys_override=None):''',
    ),
    (
        '''        for key in keys_for_level(level):''',
        '''        for key in keys_for_level(level, keys_override):''',
    ),
    # 4. make_figure propage restriction, titre et nom de sortie
    (
        '''def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax):
    accs, stds, sigs, bar_thr, thr_psdosc = collect(save_path, corrected_path, level, alpha)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    keys = keys_for_level(level)''',
        '''def make_figure(save_path, corrected_path, out_dir, level, alpha, ymin, ymax,
                keys_override=None, title=None, out_name=None):
    accs, stds, sigs, bar_thr, thr_psdosc = collect(
        save_path, corrected_path, level, alpha, keys_override
    )

    keys = keys_for_level(level, keys_override)
    # Largeur proportionnelle au nombre de barres : 14 pouces pour 9 features
    # ecrasait une figure restreinte a 4.
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(keys) + 2.0), 5.5))''',
    ),
    (
        '''    ax.set_ylabel(Y_LABEL)
    ax.set_title(
        f"Features vectorielles modernisees (subject/RFX), {LEVELS[level]}, p < {alpha}"
    )''',
        '''    ax.set_ylabel(Y_LABEL)
    ax.set_title(title if title else
                 f"Features vectorielles modernisees (subject/RFX), "
                 f"{LEVELS[level]}, p < {alpha}")''',
    ),
    (
        '''    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"barplot_vector_{level}_p{alpha}.png"''',
        '''    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (out_name if out_name
                     else f"barplot_vector_{level}_p{alpha}.png")''',
    ),
    # 5. main
    (
        '''    print(f"=== barplots vectoriels (subject/RFX, p < {args.alpha}) ===")
    for level in ("raw", "arthur", "pooled"):
        make_figure(args.save_path, args.corrected_path, args.out_dir,
                    level, args.alpha, args.ymin, args.ymax)''',
        '''    print(f"=== barplots vectoriels (subject/RFX, p < {args.alpha}) ===")
    if args.out_name and len(args.levels) > 1:
        raise SystemExit("--out-name impose un seul --levels, sinon les figures "
                         "s'ecraseraient entre elles.")
    for level in args.levels:
        make_figure(args.save_path, args.corrected_path, args.out_dir,
                    level, args.alpha, args.ymin, args.ymax,
                    args.keys, args.title, args.out_name)''',
    ),
]


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} introuvable, lancer depuis la racine du repo.")
    text = TARGET.read_text()
    changed = 0
    for old, new in REPLACEMENTS:
        if new in text:
            print(f"deja applique : {old.splitlines()[0][:62]}...")
            continue
        if text.count(old) != 1:
            raise SystemExit(
                f"motif absent ou ambigu ({text.count(old)} occurrences) :\n{old}"
            )
        text = text.replace(old, new)
        changed += 1
    if changed:
        TARGET.write_text(text)
    print(f"{changed} remplacement(s). Verifier : git diff {TARGET}")


if __name__ == "__main__":
    main()