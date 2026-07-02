"""Recalcule UNIQUEMENT les permutations (seed synchronisé, cf. BUGFIX
dans classify.py), en réutilisant les bootstraps déjà calculés cette nuit.
Évite de refaire les 1000 bootstraps (déjà valides, indépendants du bug).

Usage :
    python recompute_perms_synchronized.py \\
        --save-path /home/alouis/scratch/dream_features_noica \\
        --n-jobs $SLURM_CPUS_PER_TASK \\
        --n-perm 1000 \\
        --key cov --state S2
"""
import argparse
from pathlib import Path

import numpy as np

from classify import (
    LDA, TSclassifier, StratifiedLeave2GroupsOut, load_all,
    _run_perms_parallel, _one_perm, _result_path, _save, _clear_checkpoints,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--key", type=str, required=True)
    p.add_argument("--state", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = _result_path(args.save_path, args.key, args.state)

    if not out.exists():
        raise RuntimeError(f"{out} n'existe pas — le run initial (bootstraps) doit déjà être terminé.")

    old = np.load(out, allow_pickle=True)
    acc_scores = old["acc_scores"]  # bootstraps déjà valides, réutilisés tels quels
    acc_mean = float(acc_scores.mean()) if acc_scores.ndim == 0 or acc_scores.ndim == 1 else acc_scores.mean(axis=0)
    n_trials = int(old["n_trials"])

    print(f"=== recalcul perms (seed synchronisé) : {args.key} x {args.state} ===")
    print(f"acc_mean réutilisé (bootstraps déjà valides) : {acc_scores.mean()*100:.2f}%")

    data, labels = load_all(args.save_path, args.key, args.state)
    clf = TSclassifier(clf=LDA())
    cv = StratifiedLeave2GroupsOut()

    # Nettoie l'ancien checkpoint de perms (ancien seed, invalide pour max-stat)
    _clear_checkpoints(out)

    perm = _run_perms_parallel(
        clf, cv, data, labels, n_trials, args.n_perm,
        args.key, args.state, args.n_jobs, args.checkpoint_every, out, worker_fn=_one_perm
    )

    result = dict(old)
    result["pval"] = float((np.sum(perm >= acc_scores.mean()) + 1) / (args.n_perm + 1))
    result["perm_accs"] = perm

    _save(out, **result)
    _clear_checkpoints(out)
    print(f"nouvelle pval (non-corrigee, seed synchronise) : {result['pval']:.4f}")
