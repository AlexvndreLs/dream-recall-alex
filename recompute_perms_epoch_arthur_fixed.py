"""Permutations SCHEMA EPOCH — réplication STRICTE d'Arthur (utils.py:99-111).

DIFFERENCE avec recompute_perms_epoch_arthur.py (l'ancien) :
    L'ancien re-tirait un sous-échantillon (bootstrap_sample) A CHAQUE
    permutation via _one_perm_epoch. La loi nulle intégrait donc la variance
    du sous-échantillonnage EN PLUS de la variance de permutation, ce qui
    l'élargissait et dispersait les seuils par feature (tirets non alignés).

    Arthur (classif_cosp.py:95-107 + utils.py:99-111) tire le sous-échantillon
    UNE SEULE FOIS (prepare_data, i=0), puis permute uniquement les labels sur
    ce X FIGÉ pendant les N permutations. Nulle plus resserrée -> seuils alignés.

Ce script réplique ce protocole :
    1. bootstrap_sample UNE fois  -> X, y, groups figés
    2. score observé = CV sur ce même (X, y, groups) figé (PAS la moyenne des
       1000 bootstraps subject — cohérence observé/nulle, comme Arthur)
    3. N permutations : permute_epoch_labels(y, groups) sur le X figé

N'écrase JAMAIS le résultat subject. Écrit dans un fichier séparé, suffixe
_epochperm_fixed, même dossier results/.

  results/cov_S2.npz                   <- schéma subject (intact)
  results/cov_S2_epochperm.npz         <- ancien schéma epoch (intact)
  results/cov_S2_epochperm_fixed.npz   <- CE script (X figé, réplication stricte)

Fonctionne pour features matricielles ET vectorielles.

Usage :
    python recompute_perms_epoch_arthur_fixed.py \\
        --save-path /home/alouis/scratch/dream_features_noica_1000hz \\
        --n-jobs $SLURM_CPUS_PER_TASK \\
        --n-perm 1000 \\
        --key cov --state S2
"""
import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.classification import TSClassifier as TSclassifier

from classify import (
    StratifiedLeave2GroupsOut,
    load_all, is_matrix_feature,
    bootstrap_sample, permute_epoch_labels, run_cv, _seed, PERM_SEED_OFFSET,
    _load_checkpoint, _save_checkpoint, _clear_checkpoints,
    _result_path, _save,
)


def _epochperm_fixed_path(save_path: Path, key: str, state: str) -> Path:
    return save_path / "results" / f"{key}_{state}_epochperm_fixed.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--key", type=str, required=True)
    p.add_argument("--state", type=str, required=True)
    p.add_argument("--normalize", action="store_true", default=False)
    return p.parse_args()


def _one_perm_fixed(clf, cv, X, y, groups, key, state, p, n_perm):
    """Une permutation sur le X FIGÉ : on ne permute QUE y, groups.

    Réplique utils.py:103 d'Arthur : perm_index = permutation(len(y)) ;
    y_perm = y[perm_index] ; groups_perm = groups[perm_index]. X inchangé.
    Seed unique dépendant de p (pas de second flux de seed — l'ancien avait
    PERM_SEED_OFFSET + p pour le bootstrap ET + n_perm + p pour la perm ;
    ici plus de bootstrap dans la boucle, donc un seul seed).
    """
    y_perm, groups_perm = permute_epoch_labels(
        y, groups, _seed("perm", state, PERM_SEED_OFFSET + p)
    )
    splits = list(cv.split(X, y_perm, groups_perm))
    if X.ndim == 3:  # matrice (n_epochs, n_elec, n_elec) -> un seul classifieur global
        return run_cv(clf, splits, X, y_perm)
    # vecteur (n_epochs, n_elec) -> un LDA par électrode
    n_elec = X.shape[1]
    return np.array([run_cv(clf, splits, X[:, e:e + 1], y_perm) for e in range(n_elec)])


def _run_perms_fixed(clf, cv, X, y, groups, key, state, n_perm,
                     n_jobs, checkpoint_every, result_path):
    """Boucle de permutation sur X figé, avec checkpoint (reprise après timeout)."""
    done = _load_checkpoint(result_path, "perm")
    start = len(done) if done is not None else 0
    perms = list(done) if done is not None else []
    if start >= n_perm:
        return np.array(perms)
    if start > 0:
        print(f"    perm: reprise depuis checkpoint ({start}/{n_perm})")

    remaining = list(range(start, n_perm))
    if checkpoint_every > 0:
        for c0 in range(0, len(remaining), checkpoint_every):
            chunk = remaining[c0: c0 + checkpoint_every]
            new = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_one_perm_fixed)(clf, cv, X, y, groups, key, state, p, n_perm)
                for p in chunk
            )
            perms.extend(new)
            _save_checkpoint(result_path, "perm", np.array(perms))
            print(f"    perm: {len(perms)}/{n_perm}")
    else:
        new = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_one_perm_fixed)(clf, cv, X, y, groups, key, state, p, n_perm)
            for p in remaining
        )
        perms.extend(new)
    return np.array(perms)


if __name__ == "__main__":
    args = parse_args()

    src = _result_path(args.save_path, args.key, args.state)          # résultat subject (lecture seule, n_trials)
    dst = _epochperm_fixed_path(args.save_path, args.key, args.state)  # sortie (nouveau fichier)

    if not src.exists():
        raise RuntimeError(
            f"{src} n'existe pas — le run subject (bootstraps) doit être terminé "
            f"avant (on y lit n_trials)."
        )

    old      = np.load(src, allow_pickle=True)
    n_trials = int(old["n_trials"])
    is_matrix = is_matrix_feature(args.key)

    data, labels = load_all(args.save_path, args.key, args.state)

    # ── 1. UN seul sous-échantillon figé (comme prepare_data i=0 d'Arthur) ──
    X, y, groups = bootstrap_sample(
        data, labels, n_trials, _seed("perm", args.state, PERM_SEED_OFFSET)
    )

    if is_matrix:
        clf = TSclassifier(clf=LDA())
    else:
        clf = (Pipeline([("scaler", StandardScaler()), ("lda", LDA(solver="svd"))])
               if args.normalize else LDA(solver="svd"))
    cv = StratifiedLeave2GroupsOut()

    print(f"=== schéma EPOCH FIXÉ (réplication stricte Arthur) : {args.key} × {args.state} ===")
    print(f"X figé : {X.shape[0]} epochs (n_trials={n_trials}), {len(data)} sujets")

    # ── 2. score observé = CV sur le MÊME X figé, vrais labels (cohérence nulle) ──
    splits_obs = list(cv.split(X, y, groups))
    if is_matrix:
        acc_obs = run_cv(clf, splits_obs, X, y)
    else:
        n_elec  = X.shape[1]
        acc_obs = np.array([run_cv(clf, splits_obs, X[:, e:e + 1], y) for e in range(n_elec)])

    # ── 3. N permutations sur le X figé (permute y, groups uniquement) ──
    _clear_checkpoints(dst)
    perm = _run_perms_fixed(
        clf, cv, X, y, groups, args.key, args.state, args.n_perm,
        args.n_jobs, args.checkpoint_every, dst
    )

    # ── résultats ──
    result = dict(old)
    result["perm_scheme"] = "epoch_fixed"
    result["perm_accs"]   = perm
    result["acc_obs_fixed"] = acc_obs  # observé cohérent avec la nulle (X figé)
    if is_matrix:
        result["pval"] = float((np.sum(perm >= acc_obs) + 1) / (args.n_perm + 1))
    else:
        result["pvals"] = (np.sum(perm >= acc_obs[None, :], axis=0) + 1) / (args.n_perm + 1)

    _save(dst, **result)
    _clear_checkpoints(dst)

    if is_matrix:
        print(f"acc observé (X figé) = {float(acc_obs) * 100:.2f}%")
        print(f"pval (epoch fixé)    = {result['pval']:.4f}")
    else:
        print(f"pvals (epoch fixé) min = {result['pvals'].min():.4f}")
    print(f"Écrit : {dst}")