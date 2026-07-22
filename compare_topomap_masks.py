"""Compare les DEUX régimes de significativité d'Arthur sur nos .npz epochperm,
pour trancher lequel reproduit la Fig. 4 de la thèse (chap. 1).

Arthur a deux régimes coexistant dans visu_topomap.py :

  (A) BINOMIAL  (actif quand PREFIX = "bootstrapped_subsamp_", PERM=False) :
        seuil = binom.isf(PVAL, n_trials, 0.5) / n_trials
        electrode significative si  acc > seuil
      C'est le code EFFECTIVEMENT committé dans le dépôt public.

  (B) MAXSTAT-PERMUTATION  (actif quand PREFIX = "perm_", PERM=True) :
        pscores = max sur les 19 électrodes de la loi nulle par permutation
        pval_e  = (sum(pscores >= acc_e) + 1) / (n_perm + 1)   # compute_pval d'Arthur
        electrode significative si pval_e < PVAL
      C'est ce que dit la LÉGENDE de la Fig. 4 ("maximum statistics").

Ce script calcule les DEUX masques sur nos résultats et imprime, pour chaque
(bande × stade), le nombre d'étoiles de chaque régime, à côté du comptage lu
sur ta figure. Le régime dont le semis colle à la Fig. 4 est le bon.

Aucune écriture, aucun recalcul : lecture seule des .npz _epochperm existants.

Usage (sur le cluster, depuis le dossier du dépôt) :
    python compare_topomap_masks.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --feature-family psd \
        --pval 0.001
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binom

from config_v3 import FREQ_DICT, N_EEG, STATE_LIST

# Ordre d'affichage de la Fig. 4 : S2, SWS, NREM, REM.
STATES_DISPLAY = [s for s in ["S2", "SWS", "NREM", "REM"] if s in STATE_LIST]
BANDS = list(FREQ_DICT)  # delta, theta, alpha, sigma, beta

# Comptage d'étoiles lu sur TA Fig. 4 fournie (perm. epoch, p<0.001, corr. maxstat).
# Sert de référence visuelle pour dire quel régime colle. À corriger si je me
# suis trompé en lisant la figure.
FIG4_STARS = {
    ("delta", "S2"): 2, ("delta", "SWS"): 0, ("delta", "NREM"): 1, ("delta", "REM"): 0,
    ("theta", "S2"): 2, ("theta", "SWS"): 4, ("theta", "NREM"): 0, ("theta", "REM"): 0,
    ("alpha", "S2"): 3, ("alpha", "SWS"): 0, ("alpha", "NREM"): 1, ("alpha", "REM"): 0,
    ("sigma", "S2"): 9, ("sigma", "SWS"): 3, ("sigma", "NREM"): 6, ("sigma", "REM"): 1,
    ("beta",  "S2"): 1, ("beta",  "SWS"): 2, ("beta",  "NREM"): 1, ("beta",  "REM"): 0,
}


def compute_pval(score, perm_scores):
    """compute_pval d'Arthur (utils.py), à l'identique."""
    perm_scores = np.asarray(perm_scores)
    n_perm = len(perm_scores)
    return (np.sum(perm_scores >= score) + 1.0) / (n_perm + 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, contenant results/.")
    p.add_argument("--feature-family", default="psd", choices=["psd", "psd_osc"])
    p.add_argument("--pval", type=float, default=0.001)
    return p.parse_args()


def load_cell(save_path, family, band, state):
    """Charge acc_mean (19,), perm_accs (n_perm,19), pvals (19,), n_trials."""
    path = save_path / "results" / f"{family}_{band}_{state}_epochperm.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    acc = np.asarray(d["acc_scores"])
    acc_mean = acc.mean(axis=0) if acc.ndim == 2 else np.asarray(d["acc_mean"])
    perm = np.asarray(d["perm_accs"]) if "perm_accs" in d else None
    pvals = np.asarray(d["pvals"]) if "pvals" in d else None
    n_trials = int(d["n_trials"]) if "n_trials" in d else None
    return dict(acc_mean=acc_mean, perm=perm, pvals=pvals, n_trials=n_trials)


def mask_binomial(acc_mean, n_trials, pval):
    """Régime A : seuil binomial d'Arthur, par cellule (n_trials scalaire)."""
    if n_trials is None:
        return None, None
    thr_count = binom.isf(pval, n_trials, 0.5)
    thr_acc = thr_count / n_trials
    return acc_mean > thr_acc, thr_acc


def mask_maxstat_perm(acc_mean, perm, pval):
    """Régime B : maxstat-permutation avec compute_pval d'Arthur.

    perm : (n_perm, n_elec). La loi nulle du max est prise SUR LES ÉLECTRODES
    (axis=1), comme np.max(pscores_all_elec, axis=0) chez Arthur (son axe 0 est
    l'électrode car il empile par électrode).
    """
    if perm is None:
        return None
    null_max = perm.max(axis=1)                        # (n_perm,)
    pvalues = np.array([compute_pval(a, null_max) for a in acc_mean])
    return pvalues < pval


def main():
    args = parse_args()
    print(f"=== Comparaison des masques, famille {args.feature_family}, "
          f"p<{args.pval} ===\n")
    header = (f"{'band':>6} {'state':>5} | {'fig4':>4} | "
              f"{'binom':>5} {'thr_acc':>7} | {'maxstat':>7} | verdict")
    print(header)
    print("-" * len(header))

    tot_bin_match = 0
    tot_max_match = 0
    n_cells = 0

    for band in BANDS:
        for state in STATES_DISPLAY:
            cell = load_cell(args.save_path, args.feature_family, band, state)
            fig4 = FIG4_STARS.get((band, state), None)
            if cell is None:
                print(f"{band:>6} {state:>5} | {str(fig4):>4} | "
                      f"{'--':>5} {'--':>7} | {'--':>7} | ABSENT")
                continue

            m_bin, thr_acc = mask_binomial(cell["acc_mean"], cell["n_trials"], args.pval)
            m_max = mask_maxstat_perm(cell["acc_mean"], cell["perm"], args.pval)

            n_bin = int(m_bin.sum()) if m_bin is not None else None
            n_max = int(m_max.sum()) if m_max is not None else None

            # verdict cellule : lequel égale fig4 ?
            v = []
            if fig4 is not None:
                if n_bin == fig4:
                    v.append("BIN")
                    tot_bin_match += 1
                if n_max == fig4:
                    v.append("MAX")
                    tot_max_match += 1
                n_cells += 1
            verdict = "=".join(v) if v else ("?" if fig4 is not None else "")

            thr_str = f"{thr_acc:.4f}" if thr_acc is not None else "--"
            print(f"{band:>6} {state:>5} | {str(fig4):>4} | "
                  f"{str(n_bin):>5} {thr_str:>7} | {str(n_max):>7} | {verdict}")

    print("-" * len(header))
    print(f"\nCellules avec référence Fig.4 : {n_cells}")
    print(f"  binomial   matche Fig.4 sur : {tot_bin_match}/{n_cells} cellules")
    print(f"  maxstat    matche Fig.4 sur : {tot_max_match}/{n_cells} cellules")
    if n_cells:
        if tot_bin_match > tot_max_match:
            print("\n=> Le régime BINOMIAL colle le mieux à la Fig. 4.")
        elif tot_max_match > tot_bin_match:
            print("\n=> Le régime MAXSTAT-permutation colle le mieux à la Fig. 4.")
        else:
            print("\n=> Ex aequo ou aucun ne colle : inspecter les cellules "
                  "individuelles ci-dessus.")


if __name__ == "__main__":
    main()