#!/usr/bin/env python3
"""
Ajoute une option --keys a plot_deflation_full.py, pour restreindre
l'histogramme de deflation a une liste de descripteurs.

Usage :
    python3 patch_deflation_44.py            # applique le patch
    python3 patch_deflation_44.py --dry-run  # montre sans modifier
"""
import sys, re, shutil, os

CIBLE = "plot_deflation_full.py"

AJOUT_ARG = '''    p.add_argument("--keys", nargs="*", default=None,
                   help="Restreint aux descripteurs listes. Par defaut, tous. "
                        "Exemple : --keys cov cosp_delta cosp_theta cosp_alpha "
                        "cosp_sigma cosp_beta psd_delta psd_theta psd_alpha "
                        "psd_sigma psd_beta")
'''

FILTRE = '''
    if args.keys:
        garde = set(args.keys)
        avant = len(rows)
        rows = [r for r in rows
                if r[0].rsplit("_", 1)[0] in garde or r[0].split("_")[0] in garde]
        print(f"  filtre --keys : {avant} -> {len(rows)} combinaisons")
'''

def main():
    dry = "--dry-run" in sys.argv
    if not os.path.exists(CIBLE):
        sys.exit(f"{CIBLE} introuvable. Lancer depuis la racine du depot.")
    src = open(CIBLE).read()

    if "--keys" in src:
        sys.exit("Le patch semble deja applique.")

    ancre_arg = '    args = p.parse_args()'
    assert ancre_arg in src, "ancre argparse introuvable"
    src = src.replace(ancre_arg, AJOUT_ARG + ancre_arg, 1)

    ancre_rows = '    combos = [r[0] for r in rows]'
    assert ancre_rows in src, "ancre rows introuvable"
    src = src.replace(ancre_rows, FILTRE + "\n" + ancre_rows, 1)

    if dry:
        print(src[:2000]); return
    shutil.copy(CIBLE, CIBLE + ".bak")
    open(CIBLE, "w").write(src)
    print(f"Patch applique. Sauvegarde : {CIBLE}.bak")

main()