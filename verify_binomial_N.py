"""Vérifie empiriquement quel N binomial reproduit le semis d'étoiles de la
Fig. 4 d'Arthur (chap. 1).

Le masque d'étoiles d'Arthur (code committé, visu_topomap.py, régime BINOM) est :
    seuil_acc = binom.isf(PVAL, N, 0.5) / N
    electrode significative si  acc > seuil_acc
où N = INFO_DATA.iloc[36] par stade. On ne dispose pas de son info_data.csv,
donc on TESTE plusieurs candidats de N et on regarde lequel matche la Fig. 4 :

  - 61   : n_trials (épochs équilibrées par sujet, notre .npz)         -> seuil ~68.9%
  - 122  : 2 sujets x 61 (épochs de test par split de la CV LPGO P=2)  -> seuil ~63.9%
  - POOL : pool total d'épochs de test par stade (depuis le xlsx)       -> seuil ~51-52%

Ce script NE change QUE le masque d'étoiles. Le fond coloré des topomaps
(acc_mean par électrode) est identique quel que soit N : il n'est jamais touché.

Lecture seule des .npz existants, aucune écriture, aucun recalcul.

Usage (sur le cluster, depuis le dossier du dépôt) :
    python verify_binomial_N.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --feature-family psd \
        --pval 0.001
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binom

from config_v3 import FREQ_DICT, STATE_LIST

STATES_DISPLAY = [s for s in ["S2", "SWS", "NREM", "REM"] if s in STATE_LIST]
BANDS = list(FREQ_DICT)

# Pool total d'épochs par stade (2 classes réunies), lu depuis
# Riemannian_Dream_Recall_Subject_numbers.xlsx (colonne Total).
POOL_TOTAL = {"S2": 10310, "SWS": 8758, "REM": 5887, "NREM": 21588}

# n_trials équilibré par sujet (min inter-sujets global = sujet 38 en REM).
# Constant sur tous les stades car c'est le MIN GLOBAL, comme Arthur.
N_TRIALS_BALANCED = 61
N_TEST_PER_SPLIT = 2 * N_TRIALS_BALANCED  # 122 : 1 sujet HR + 1 LR en test

# Ton comptage d'étoiles recompté sur la Fig. 4 (nouveau rendu viridis).
# À corriger si erreur de lecture.
FIG4_STARS = {
    ("delta", "S2"): 2, ("delta", "SWS"): 0, ("delta", "NREM"): 1, ("delta", "REM"): 0,
    ("theta", "S2"): 2, ("theta", "SWS"): 0, ("theta", "NREM"): 0, ("theta", "REM"): 0,
    ("alpha", "S2"): 3, ("alpha", "SWS"): 0, ("alpha", "NREM"): 0, ("alpha", "REM"): 0,
    ("sigma", "S2"): 9, ("sigma", "SWS"): 2, ("sigma", "NREM"): 1, ("sigma", "REM"): 0,
    ("beta",  "S2"): 1, ("beta",  "SWS"): 2, ("beta",  "NREM"): 1, ("beta",  "REM"): 0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--feature-family", default="psd", choices=["psd", "psd_osc"])
    p.add_argument("--pval", type=float, default=0.001)
    return p.parse_args()


def load_acc(save_path, family, band, state):
    """Charge acc_mean (vecteur de 19 accuracies par électrode)."""
    path = save_path / "results" / f"{family}_{band}_{state}_epochperm.npz"
    if not path.exists():
        # fallback schéma subject si epochperm absent (les acc sont les mêmes,
        # seules les perms diffèrent, et le binomial n'utilise pas les perms)
        path = save_path / "results" / f"{family}_{band}_{state}.npz"
        if not path.exists():
            return None
    d = np.load(path, allow_pickle=True)
    acc = np.asarray(d["acc_scores"])
    return acc.mean(axis=0) if acc.ndim == 2 else np.asarray(d["acc_mean"])


def binom_threshold(N, pval):
    """Seuil d'accuracy d'Arthur : binom.isf(pval, N, 0.5) / N."""
    return binom.isf(pval, N, 0.5) / N


def main():
    args = parse_args()
    pval = args.pval

    # Seuils par candidat. Pour 61 et 122 : constant. Pour POOL : par stade.
    thr61 = binom_threshold(N_TRIALS_BALANCED, pval)
    thr122 = binom_threshold(N_TEST_PER_SPLIT, pval)
    thr_pool = {st: binom_threshold(POOL_TOTAL[st], pval) for st in STATES_DISPLAY}

    print(f"=== Vérification N binomial, famille {args.feature_family}, "
          f"p<{pval} ===\n")
    print(f"Seuils d'accuracy par candidat :")
    print(f"  N=61   (épochs/sujet)        -> {thr61*100:.2f}%  (constant)")
    print(f"  N=122  (test par split)      -> {thr122*100:.2f}%  (constant)")
    print(f"  N=pool (total test par stade):")
    for st in STATES_DISPLAY:
        print(f"       {st:5s} N={POOL_TOTAL[st]:6d} -> {thr_pool[st]*100:.2f}%")
    print()

    header = (f"{'band':>6} {'state':>5} | {'fig4':>4} | "
              f"{'N=61':>5} {'N=122':>6} {'N=pool':>7} | verdict")
    print(header)
    print("-" * len(header))

    match = {"61": 0, "122": 0, "pool": 0}
    n_cells = 0

    for band in BANDS:
        for state in STATES_DISPLAY:
            acc = load_acc(args.save_path, args.feature_family, band, state)
            fig4 = FIG4_STARS.get((band, state), None)
            if acc is None:
                print(f"{band:>6} {state:>5} | {str(fig4):>4} | "
                      f"{'--':>5} {'--':>6} {'--':>7} | ABSENT")
                continue

            n61 = int((acc > thr61).sum())
            n122 = int((acc > thr122).sum())
            npool = int((acc > thr_pool[state]).sum())

            v = []
            if fig4 is not None:
                n_cells += 1
                if n61 == fig4:
                    v.append("61"); match["61"] += 1
                if n122 == fig4:
                    v.append("122"); match["122"] += 1
                if npool == fig4:
                    v.append("pool"); match["pool"] += 1
            verdict = "=".join(v) if v else ("?" if fig4 is not None else "")

            print(f"{band:>6} {state:>5} | {str(fig4):>4} | "
                  f"{n61:>5} {n122:>6} {npool:>7} | {verdict}")

    print("-" * len(header))
    print(f"\nCellules avec référence Fig.4 : {n_cells}")
    for cand in ["61", "122", "pool"]:
        print(f"  N={cand:>4} matche Fig.4 sur : {match[cand]}/{n_cells} cellules")

    best = max(match, key=match.get)
    print(f"\n=> Meilleur candidat : N={best} "
          f"({match[best]}/{n_cells} cellules matchées)")
    if match[best] == 0:
        print("   (aucun ne matche : le N binomial est encore ailleurs, "
              "ou le comptage Fig.4 est à revoir)")


if __name__ == "__main__":
    main()