#!/usr/bin/env python3
"""Colonne 3 de la Fig.3 : t-test avec le SCHEMA DE PERMUTATION D'ARTHUR.

Script AUTONOME. Ne modifie pas recompute_ttest_fig3.py, il en importe les
fonctions de chargement et de preparation des donnees, donc le chemin de donnees
est strictement le meme. La sortie a le meme format npz, donc
plot_fig3_arthur_topomaps.py fonctionne sans changement.

Ce que ce script ajoute : --scheme lexico.
-----------------------------------------
Un test de permutation redistribue les etiquettes AU HASARD. Arthur, lui, appelle
itertools.combinations (_combinations, ttest_perm_indep.py l.303-310), qui n'est
pas un melangeur mais un odometre : il enumere les sous-ensembles dans l'ordre
lexicographique, en incrementant d'abord le dernier indice. Aucun generateur
pseudo-aleatoire n'est appele nulle part dans son fichier.

  1. La premiere combinaison (0, ..., n_cond1-1) redonne EXACTEMENT cond1, puisque
     full_mat = concatenate((cond1, cond2)) l.152. Il la retire l.180
     ("the first perm is not a permutation"). Reproduit ici.
  2. Les suivantes n'en different que par une ou deux positions. Au niveau epoch
     (n_cond1 ~ 5000), chaque "permutation" deplace moins de 0,05 % du groupe. La
     distribution nulle n'est pas centree sur zero, elle est centree sur la
     statistique observee.
  3. Le denominateur de la p-value vaut len(perm_t) APRES le retrait, soit
     n_perm - 1 (compute_pvalues l.201). Reproduit ici.

--zscore none, et surtout PAS --zscore subject
---------------------------------------------
Un z-score par sujet recentre chaque sujet a moyenne exactement nulle, donc les
moyennes de groupe s'annulent et le t de Welch tombe a ~1e-14, du bruit d'arrondi.
Mesure sur nos donnees : max|t| = 6.1e-14. Sa figure publiee montre au contraire
des t-values spatialement structurees dans une plage de z ordinaire (-2 a +1), ce
qui exclut ce chemin. Son visu_topomap.py lit d'ailleurs dans psd/results, pas
dans zscore_psd/results ou ttest.py ecrit.

Verification sans donnees
-------------------------
    python recompute_ttest_fig3_lexico.py --selftest

Usage reel
----------
    python recompute_ttest_fig3_lexico.py \\
        --save-path /scratch/alouis/dream_features_noica_1000hz \\
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_permscheme_arthur \\
        --state S2 --n-perm 9999 --level epoch --zscore none \\
        --maxstat-scope electrodes --drop-subjects 10 \\
        --scheme lexico --arthur-pval-bug --n-jobs 32
"""

import argparse
import sys
from itertools import combinations, islice
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from config_v3 import N_EEG
from recompute_ttest_fig3 import (
    BANDS,
    _ttest_perm,
    build_conditions,
    load_subject_epochs,
)


# --------------------------------------------------------------- schemas
def perm_indices_lexico(n_samples, n_cond1, n_perm):
    """Les n_perm premieres combinaisons lexicographiques, moins la premiere.

    Renvoie un GENERATEUR : materialiser 9999 tuples de ~5000 entiers coute
    plusieurs Go. joblib.Parallel consomme un generateur sans probleme.
    Les indices sont convertis en int64 pour que le pickling vers les workers
    reste leger.
    """
    gen = islice(combinations(range(n_samples), n_cond1), n_perm)
    gen = (np.fromiter(c, dtype=np.int64, count=n_cond1) for c in gen)
    next(gen, None)          # Arthur l.180 : la 1ere combinaison est l'identite
    return gen


def perm_indices_random(n_samples, n_cond1, n_perm, seed):
    """Tirage aleatoire uniforme sans remise. Le schema correct, pour reference."""
    rng = np.random.RandomState(seed)
    return (rng.choice(n_samples, size=n_cond1, replace=False) for _ in range(n_perm))


def ttest_maxstat(cond1, cond2, n_perm, seed, n_jobs, scheme, arthur_pval_bug):
    """t de Welch par electrode, correction par statistique du maximum.

    arthur_pval_bug=False : two-tailed symetrique, |t_obs| compare au max de
        |t_perm|. Un effet HR>LR et un effet HR<LR sont traites pareil.

    arthur_pval_bug=True : reproduit le biais de signe de compute_pvalues
        (ttest_perm_indep.py l.204 et l.210-215). abs() est applique a la nulle
        mais pas a l'observe, et la comparaison est tstat <= t_perm. Pour une
        electrode a effet negatif la condition est vraie pour presque toutes les
        permutations, donc p ~ 1 : son two-tailed se comporte en unilateral HR>LR.
    """
    tval = ttest_ind(cond1, cond2, equal_var=False)[0]
    full = np.vstack((cond1, cond2))
    if scheme == "lexico":
        idxs = perm_indices_lexico(len(full), len(cond1), n_perm)
    else:
        idxs = perm_indices_random(len(full), len(cond1), n_perm, seed)

    perm_t = Parallel(n_jobs=n_jobs)(delayed(_ttest_perm)(full, ix) for ix in idxs)
    perm_t = np.asarray(perm_t)
    # scaling = len(perm_t) APRES retrait eventuel, comme Arthur l.201.
    # lexico -> n_perm - 1 ; random -> n_perm.
    scaling = len(perm_t)

    perm_max = np.abs(perm_t).max(axis=1)
    obs = tval if arthur_pval_bug else np.abs(tval)
    num = (perm_max[:, None] >= obs[None, :]).sum(axis=0).astype(float)
    return tval, num / scaling, perm_max


# -------------------------------------------------------------- selftest
def selftest():
    """Verifie le comportement des deux schemas sur donnees synthetiques.

    Effet reel de 1 ecart-type, 500 observations par groupe, 19 canaux.
    Attendu : la nulle lexico colle a l'observe, la nulle aleatoire est autour
    de la valeur attendue d'un maximum sur 19 canaux.
    """
    rng = np.random.RandomState(0)
    n1 = n2 = 500
    c1 = rng.randn(n1, 19) + 1.0
    c2 = rng.randn(n2, 19)

    print("selftest : effet reel de 1 sd, 500 obs par groupe, 19 canaux, 200 perms\n")
    for scheme in ("lexico", "random"):
        t, p, pmax = ttest_maxstat(c1, c2, 200, 0, 1, scheme, False)
        print(f"  {scheme:7s} : max|t| observe = {np.abs(t).max():6.2f}   "
              f"nulle moy = {pmax.mean():6.2f}  sd = {pmax.std():5.2f}   "
              f"p_min = {p.min():.3f}")
    print("\nAttendu : nulle lexico collee a l'observe et p_min elevee, "
          "nulle aleatoire basse et p_min a 0.")


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--selftest", action="store_true",
                   help="Verifie les deux schemas sur donnees synthetiques, "
                        "sans toucher aux donnees reelles. Sort ensuite.")
    p.add_argument("--save-path", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--state", type=str, default="S2")
    p.add_argument("--n-perm", type=int, default=9999,
                   help="9999, valeur de ttest.py l.14.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--level", choices=["epoch", "subject"], default="epoch")
    p.add_argument("--zscore", choices=["none", "global", "subject"], default="none",
                   help="'none' par defaut. 'subject' annule l'effet de groupe "
                        "(t ~ 1e-14, mesure), a n'utiliser pour rien.")
    p.add_argument("--maxstat-scope", choices=["electrodes", "both"],
                   default="electrodes")
    p.add_argument("--drop-subjects", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scheme", choices=["lexico", "random"], default="lexico",
                   help="'lexico' = schema d'Arthur, replique exacte. 'random' = "
                        "tirage aleatoire uniforme, le schema correct, fourni "
                        "comme point de comparaison.")
    p.add_argument("--arthur-pval-bug", action="store_true", default=False,
                   help="Reproduit le biais de signe two-tailed (E6).")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if args.save_path is None or args.out_dir is None:
        sys.exit("--save-path et --out-dir sont requis (ou utilise --selftest).")

    t0 = time()
    out = args.out_dir / f"fig3_ttest_{args.state}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    drop_ids = {int(s.strip()) for s in args.drop_subjects.split(",") if s.strip()}
    per_band_epochs, labels = load_subject_epochs(args.save_path, args.state, drop_ids)
    n_hr = int((labels == 1).sum())
    n_lr = int((labels == 0).sum())
    print(f"[{args.state}] sujets : {len(labels)} (HR={n_hr}, LR={n_lr}) | "
          f"level={args.level} | zscore={args.zscore} | scheme={args.scheme} | "
          f"n_perm={args.n_perm} | drop={sorted(drop_ids) or 'aucun'}")
    if n_hr < 2 or n_lr < 2:
        raise RuntimeError("Pas assez de sujets par groupe.")

    conds = build_conditions(per_band_epochs, labels, args.level, args.zscore)
    if args.level == "epoch":
        n1 = conds[BANDS[0]][0].shape[0]
        n2 = conds[BANDS[0]][1].shape[0]
        print(f"  niveau epoch : {n1} epochs HR vs {n2} epochs LR (n total={n1+n2})")
        if args.scheme == "lexico":
            print(f"  [MODE ARTHUR] lexico : les {args.n_perm} permutations ne "
                  f"deplacent qu'une ou deux epochs sur {n1} "
                  f"({100.0 / n1:.3f} % du groupe). Nulle degeneree, PAS un resultat.")
    if args.zscore == "subject":
        print("  ATTENTION : --zscore subject annule l'effet de groupe, "
              "les t seront du bruit d'arrondi (~1e-14).")
    if args.arthur_pval_bug:
        print("  [MODE ARTHUR] biais de signe two-tailed actif (E6) : "
              "effets HR<LR invisibles.")

    tvals, pvals, nulls = {}, {}, {}
    if args.maxstat_scope == "electrodes":
        for b in BANDS:
            c1, c2 = conds[b]
            tv, pv, pm = ttest_maxstat(c1, c2, args.n_perm, args.seed, args.n_jobs,
                                       args.scheme, args.arthur_pval_bug)
            tvals[b], pvals[b], nulls[b] = tv, pv, pm
    else:
        c1 = np.concatenate([conds[b][0] for b in BANDS], axis=1)
        c2 = np.concatenate([conds[b][1] for b in BANDS], axis=1)
        tv, pv, pm = ttest_maxstat(c1, c2, args.n_perm, args.seed, args.n_jobs,
                                   args.scheme, args.arthur_pval_bug)
        for i, b in enumerate(BANDS):
            tvals[b] = tv[i * N_EEG:(i + 1) * N_EEG]
            pvals[b] = pv[i * N_EEG:(i + 1) * N_EEG]
            nulls[b] = pm

    print(f"\n=== {args.state}, scheme={args.scheme}, "
          f"pval_bug={args.arthur_pval_bug} ===")
    print(f"{'bande':8s} {'max|t|':>8s} {'nulle moy':>10s} {'nulle sd':>9s} "
          f"{'p_min':>7s} {'n(p<.001)':>10s}")
    for b in BANDS:
        print(f"{b:8s} {np.abs(tvals[b]).max():8.2f} {nulls[b].mean():10.2f} "
              f"{nulls[b].std():9.2f} {pvals[b].min():7.4f} "
              f"{int((pvals[b] < 0.001).sum()):7d}/19")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        bands=np.array(BANDS),
        tvals=np.array([tvals[b] for b in BANDS]),
        pvals=np.array([pvals[b] for b in BANDS]),
        null_mean=np.array([nulls[b].mean() for b in BANDS]),
        null_std=np.array([nulls[b].std() for b in BANDS]),
        labels=labels,
        n_hr=n_hr, n_lr=n_lr,
        state=args.state,
        n_perm=args.n_perm,
        level=args.level,
        zscore=args.zscore,
        maxstat_scope=args.maxstat_scope,
        drop_subjects=sorted(drop_ids),
        two_tailed=True,
        perm_scheme=args.scheme,
        arthur_pval_bug=args.arthur_pval_bug,
    )
    print(f"\nSauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()
