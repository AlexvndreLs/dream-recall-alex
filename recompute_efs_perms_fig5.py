"""Recompute Fig. 5 (these Arthur chap.1) - SCRIPT 2/2 : permutations p<0.001.

REPLIQUE EXACTE d'Arthur (permutations_EFS_fixed_elec.py). Recharge le resultat de
l'EFS holdout (script 1, efs_holdout_{state}_{elec}.npz), et pour chaque split ajoute
des permutations de labels au niveau epoch sur les bandes DEJA selectionnees (pas de
re-EFS, comme Arthur). Produit une p-value par split.

============================ FIDELITE : BUGS D'ARTHUR REPLIQUES ================
Sur demande (fidelite exacte, corrections pour plus tard), on reproduit tel quel le
comportement de permutations_EFS_fixed_elec.py, Y COMPRIS ce qui ressemble a des bugs.
Verifie ligne par ligne dans son code :

1. pscores NON REINITIALISE entre splits (son code : `pscores = []` est AVANT la boucle
   des splits, jamais remis a zero dedans). Donc au split i, compute_pval compare le
   score du split i a TOUTES les permutations des splits 0..i cumulees (1000, puis
   2000, ... puis 324000 pscores). Presque certainement non intentionnel, mais
   REPLIQUE ici (flag --arthur-cumulative-pscores, defaut True). Passer
   --no-arthur-cumulative-pscores pour la version propre (pscores par split).

2. Permutation INDEPENDANTE de y_feature (train) et y_classif (test) : il permute les
   deux separement (np.random.permutation appele 2x). Replique.

3. compute_pval = (sum(perm >= score) + 1) / (n_perm + 1), avec n_perm = len(pscores)
   au moment de l'appel (donc croissant si cumulative). Replique.

Ces choix sont DOCUMENTES pour correction ulterieure, pas corriges maintenant.

Difference de RNG (inevitable) : Arthur permute en serie avec un np.random global
non seede. Ici on parallelise (prefer='processes') avec un seed par split (seed+i),
donc permutations REPRODUCTIBLES mais pas bit-identiques aux siennes. Le resultat
statistique est equivalent (1000 permutations aleatoires des labels par split).
================================================================================

N_PERM = 1000 par split (valeur d'Arthur). p minimum ~ 1/1001 = 0.001, pile au seuil
p<0.001 de la Fig.5.

Entrees : {out_dir}/efs_holdout_{state}_{elec}.npz (script 1) + features psd_{band}.
Sorties : {out_dir}/efs_perms_{state}_{elec}.npz
          pvalues (324,), pscores (liste cumulative finale), best_freqs, test_scores.

Usage (une electrode, pour job array)
-------------------------------------
    python recompute_efs_perms_fig5.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_corrected/fig5_recompute \
        --state     S2 --elec Fz --n-perm 1000 --n-jobs $SLURM_CPUS_PER_TASK

Author: recompute pour Alex (replique Arthur chap.1, Fig.5 permutations)
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from config_v3 import (
    FREQ_DICT,
    CH_NAMES,
    N_EEG,
    SUBJECT_LIST_ORDERED,
    SUBJECT_LABELS,
    CLASSIFICATION_GROUPS,
)
from utils import load_atomic
from classify import StratifiedLeave2GroupsOut

BANDS = list(FREQ_DICT)
EEG_CH = CH_NAMES[:N_EEG]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--state",     type=str, default="S2")
    p.add_argument("--elec",      type=str, required=True,
                   help="Electrode a traiter (ex 'Fz'). Job array = 1 elec/tache.")
    p.add_argument("--n-perm",    type=int, default=1000,
                   help="1000 = valeur d'Arthur.")
    p.add_argument("--n-jobs",    type=int, default=1)
    p.add_argument("--arthur-cumulative-pscores", dest="cumulative",
                   action="store_true", default=True,
                   help="REPLIQUE Arthur : pscores accumule entre splits (defaut).")
    p.add_argument("--no-arthur-cumulative-pscores", dest="cumulative",
                   action="store_false",
                   help="Version propre : pscores reinitialise par split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def load_electrode_subs(save_path, state, elec_idx):
    """Retourne subs : liste (un array (n_epochs, 5) par sujet) pour l'electrode + labels.

    Meme chargement que le script 1, slice sur l'electrode. Toutes epochs (no subsampling).
    """
    stages = CLASSIFICATION_GROUPS[state]
    subs, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        band_arrays = []
        ok = True
        for b in BANDS:
            parts = [a for s in stages
                     if (a := load_atomic(save_path, f"psd_{b}", sub_id, s)) is not None]
            if not parts:
                ok = False
                break
            band_arrays.append(np.concatenate(parts, axis=0)[:, elec_idx])  # (n_epochs,)
        if not ok:
            continue
        subs.append(np.stack(band_arrays, axis=1))    # (n_epochs, 5)
        labels.append(label)
    return subs, np.array(labels)


def compute_pval(score, perm_scores):
    """(sum(perm >= score) + 1) / (n_perm + 1). Replique compute_pval d'Arthur (utils)."""
    perm = np.asarray(perm_scores)
    return (np.sum(perm >= score) + 1.0) / (len(perm) + 1.0)


def permute_split(x_feature, y_feature, x_classif, y_classif, best_idx, n_perm, seed):
    """n_perm scores de permutation pour UN split (bandes best_idx deja fixees).

    Replique Arthur : permute y_feature ET y_classif independamment, refit LDA sur
    best_idx, score. Retourne la liste des n_perm pscores de CE split.
    """
    rng = np.random.RandomState(seed)
    out = []
    xf = x_feature[:, best_idx]
    xc = x_classif[:, best_idx]
    for _ in range(n_perm):
        yf = rng.permutation(y_feature)
        yc = rng.permutation(y_classif)
        clf = LDA()
        clf.fit(xf, yf)
        out.append(float(clf.score(xc, yc)))
    return out


def main():
    args = parse_args()
    t0 = time()

    if args.elec not in EEG_CH:
        raise ValueError(f"{args.elec} pas dans {EEG_CH}")
    elec_idx = EEG_CH.index(args.elec)

    out = args.out_dir / f"efs_perms_{args.state}_{args.elec}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    # recharge le resultat EFS holdout du script 1 (best_freqs + test_scores)
    efs_file = args.out_dir / f"efs_holdout_{args.state}_{args.elec}.npz"
    if not efs_file.exists():
        raise FileNotFoundError(
            f"{efs_file} absent : lancer d'abord le script 1 (EFS holdout) pour {args.elec}.")
    efs = np.load(efs_file, allow_pickle=True)
    # script 1 empile par electrode : ici 1 seule electrode -> index 0
    best_freqs = list(efs["best_freqs"][0])          # 324 sous-ensembles de bandes
    test_scores = np.asarray(efs["test_scores"][0])  # (324,)

    subs, labels = load_electrode_subs(args.save_path, args.state, elec_idx)
    print(f"[{args.state}/{args.elec}] {len(labels)} sujets, {len(best_freqs)} splits, "
          f"n_perm={args.n_perm}, cumulative={args.cumulative}")

    groups_all = np.arange(len(subs))
    splits = list(StratifiedLeave2GroupsOut().split(subs, labels, groups_all))

    # pre-calcule, par split, les matrices train/test et best_idx
    split_payloads = []
    for i, (tr, te) in enumerate(splits):
        x_feature = np.concatenate([subs[j] for j in tr], axis=0)   # (n_ep_tr, 5)
        y_feature = np.concatenate([[labels[j]] * len(subs[j]) for j in tr])
        x_classif = np.concatenate([subs[j] for j in te], axis=0)   # (n_ep_te, 5)
        y_classif = np.concatenate([[labels[j]] * len(subs[j]) for j in te])
        best_idx = [BANDS.index(b) for b in best_freqs[i]]
        split_payloads.append((x_feature, y_feature, x_classif, y_classif, best_idx))

    # permutations par split, en parallele (prefer=processes, cf EPYC)
    per_split_pscores = Parallel(n_jobs=args.n_jobs, prefer="processes")(
        delayed(permute_split)(xf, yf, xc, yc, bi, args.n_perm, args.seed + i)
        for i, (xf, yf, xc, yc, bi) in enumerate(split_payloads)
    )

    # calcul des p-values, en respectant le comportement d'Arthur
    pvalues = []
    if args.cumulative:
        # REPLIQUE ARTHUR : pscores accumule ; p-value du split i comparee a tous les
        # pscores des splits 0..i cumules.
        pool = []
        for i, ps in enumerate(per_split_pscores):
            pool.extend(ps)
            pvalues.append(compute_pval(test_scores[i], pool))
        final_pscores = pool
    else:
        # version propre : p-value du split i sur ses SEULS pscores.
        for i, ps in enumerate(per_split_pscores):
            pvalues.append(compute_pval(test_scores[i], ps))
        final_pscores = [p for ps in per_split_pscores for p in ps]

    pvalues = np.asarray(pvalues)
    n_sig = int((pvalues < 0.001).sum())
    print(f"  splits p<0.001 : {n_sig}/{len(pvalues)}  | "
          f"p median = {np.median(pvalues):.4f}  | acc holdout = {test_scores.mean():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        electrode=args.elec,
        pvalues=pvalues,                          # (324,)
        test_scores=test_scores,                  # (324,)
        best_freqs=np.array(best_freqs, dtype=object),
        pscores=np.asarray(final_pscores),
        n_perm=args.n_perm,
        cumulative=args.cumulative,
        state=args.state,
    )
    print(f"Sauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()
