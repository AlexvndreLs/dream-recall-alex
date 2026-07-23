"""Verifie si le bug de signe d'Arthur (E6) a un effet observable sur nos donnees.

Le bug ne peut se manifester QUE sur les electrodes ou t_obs < 0 (effet HR < LR).
Pour celles-la, la p d'Arthur est artificiellement poussee vers 1. Sur les
electrodes a t_obs > 0, les deux methodes sont identiques par construction.

Ce script repond a trois questions :
  1. Combien d'electrodes ont t_obs < 0, et avec quelle magnitude ?
  2. Sur celles-la, de combien la p change-t-elle entre les deux versions ?
  3. Le changement fait-il franchir un seuil a une electrode (p<0.05 ou p<0.001) ?

Si la reponse a (3) est non, le bug est reel dans le code mais SANS CONSEQUENCE
sur ce jeu de donnees, et il faut le presenter comme tel.
"""

import argparse
from pathlib import Path

import numpy as np

# Ordre des electrodes tel que produit par le pipeline (19 canaux EEG).
try:
    from config_v3 import CHANNEL_NAMES
    CH = list(CHANNEL_NAMES)
except Exception:
    CH = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--correct", type=Path, required=True,
                   help="npz produit SANS --arthur-pval-bug")
    p.add_argument("--arthur", type=Path, required=True,
                   help="npz produit AVEC --arthur-pval-bug")
    p.add_argument("--t-threshold", type=float, default=1.5,
                   help="magnitude a partir de laquelle un t negatif est juge notable")
    return p.parse_args()


def main():
    args = parse_args()
    dc = np.load(args.correct, allow_pickle=True)
    da = np.load(args.arthur, allow_pickle=True)

    bands = [str(b) for b in dc["bands"]]
    tc, pc = dc["tvals"], dc["pvals"]      # (5, 19)
    ta, pa = da["tvals"], da["pvals"]

    # Garde-fou : les t observes doivent etre IDENTIQUES entre les deux runs.
    # Seule la p change (le bug porte sur la comparaison, pas sur la statistique).
    if not np.allclose(tc, ta, rtol=1e-9, atol=1e-12):
        raise RuntimeError(
            "Les t-values different entre les deux npz. Ce ne sont pas deux runs "
            "du meme calcul, la comparaison n'a pas de sens. Verifie que seul le "
            "flag --arthur-pval-bug differe entre les deux commandes."
        )

    ch = CH if CH is not None and len(CH) == tc.shape[1] else \
        [f"e{i:02d}" for i in range(tc.shape[1])]

    n_neg = int((tc < 0).sum())
    n_tot = tc.size
    print("=== 1. Electrodes a effet negatif (HR < LR) ===")
    print(f"  {n_neg}/{n_tot} couples (bande, electrode) ont t_obs < 0")
    if n_neg == 0:
        print("\n  Aucun t negatif. Le bug de signe est LITTERALEMENT inobservable")
        print("  sur ces donnees : il ne peut affecter que les t negatifs.")
        return
    print(f"  t le plus negatif : {tc.min():+.3f}")
    n_notable = int((tc < -args.t_threshold).sum())
    print(f"  dont {n_notable} avec |t| > {args.t_threshold}")

    print("\n=== 2. Effet du bug sur les p des electrodes negatives ===")
    print(f"{'bande':7s} {'elec':6s} {'t_obs':>8s} {'p_correct':>10s} "
          f"{'p_arthur':>10s} {'delta':>9s}")
    rows = []
    for i, b in enumerate(bands):
        for j in range(tc.shape[1]):
            if tc[i, j] < 0:
                rows.append((abs(tc[i, j]), b, ch[j], tc[i, j],
                             pc[i, j], pa[i, j]))
    rows.sort(reverse=True)
    for _, b, e, t, p1, p2 in rows[:15]:
        print(f"{b:7s} {e:6s} {t:+8.3f} {p1:10.4f} {p2:10.4f} {p2 - p1:+9.4f}")
    if len(rows) > 15:
        print(f"  ... ({len(rows) - 15} autres, magnitudes plus faibles)")

    print("\n=== 3. Franchissement de seuil ===")
    neg = tc < 0
    for alpha in (0.05, 0.001):
        sig_c = int((neg & (pc < alpha)).sum())
        sig_a = int((neg & (pa < alpha)).sum())
        print(f"  p < {alpha:<6g} : {sig_c} electrode(s) negative(s) significative(s) "
              f"en version correcte, {sig_a} en version Arthur")
        if sig_c > sig_a:
            print(f"    -> le bug MASQUE {sig_c - sig_a} effet(s) reel(s) a ce seuil.")
        elif sig_c == 0:
            print("    -> aucun effet negatif ne franchit ce seuil de toute facon, "
                  "le bug ne masque rien d'observable ici.")

    pmax_neg = pa[neg].max()
    print(f"\n  p maximale atteinte par une electrode negative en version Arthur : "
          f"{pmax_neg:.4f}")
    print("  (proche de 1 = signature du bug : la p est poussee vers 1 par "
          "construction)")


if __name__ == "__main__":
    main()
