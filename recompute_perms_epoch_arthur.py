"""Calcule les permutations SCHEMA EPOCH (réplique EXACTE d'Arthur, utils.py:103
du repo arthurdehgan/sleep), en réutilisant les bootstraps déjà calculés par
classify.py (schéma subject) — aucun recalcul des 1000 bootstraps.

Contrairement à recompute_perms_synchronized.py (qui corrige un bug de seed et
ÉCRASE le résultat en place), ce script n'écrase JAMAIS le résultat subject :
il écrit dans un fichier séparé, même dossier results/, suffixe _epochperm
(cf Option 3 : rapporter les deux schémas côte à côte).

  results/cov_S2.npz            <- schéma subject (inchangé, jamais touché)
  results/cov_S2_epochperm.npz  <- schéma epoch (ce script), copie de l'ancien
                                    + perm_accs/pval recalculés avec _one_perm_epoch

Pré-requis : results/{key}_{state}.npz doit déjà exister (run --perm-scheme
subject terminé, cf classify.py). Fonctionne pour features matricielles ET
vectorielles (is_matrix_feature détermine le worker à utiliser).

Usage :
    python recompute_perms_epoch_arthur.py \\
        --save-path /home/alouis/scratch/dream_features_noica_1000hz \\
        --n-jobs $SLURM_CPUS_PER_TASK \\
        --n-perm 1000 \\
        --key cov --state S2
"""
import argparse
from pathlib import Path

import numpy as np

from classify import (
    LDA, TSclassifier, Pipeline, StandardScaler, StratifiedLeave2GroupsOut,
    load_all, is_matrix_feature, _run_perms_parallel,
    _one_perm_epoch, _one_perm_epoch_vector,
    _result_path, _save, _clear_checkpoints,
)


def _epochperm_path(save_path: Path, key: str, state: str) -> Path:
    """Même dossier results/ que _result_path (classify.py), suffixe _epochperm
    pour ne JAMAIS écraser le résultat schéma subject déjà calculé."""
    return save_path / "results" / f"{key}_{state}_epochperm.npz"
# Construit le chemin du fichier de sortie schéma epoch, distinct du fichier subject dans le même dossier results/.


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


if __name__ == "__main__":
    args = parse_args()

    src = _result_path(args.save_path, args.key, args.state)      # résultat subject existant (LECTURE SEULE)
    dst = _epochperm_path(args.save_path, args.key, args.state)   # sortie schéma epoch (nouveau fichier)

    if not src.exists():
        raise RuntimeError(
            f"{src} n'existe pas — le run initial --perm-scheme subject "
            f"(bootstraps) doit déjà être terminé avant de lancer le schéma epoch."
        )

    old = np.load(src, allow_pickle=True)
    acc_scores = old["acc_scores"]          # bootstraps déjà valides, réutilisés tels quels
    n_trials   = int(old["n_trials"])
    is_matrix  = is_matrix_feature(args.key)

    print(f"=== schéma EPOCH (Arthur, utils.py:103) : {args.key} × {args.state} ===")
    print(f"acc_scores réutilisés depuis {src.name} (pas de recalcul bootstrap)")

    data, labels = load_all(args.save_path, args.key, args.state)

    if is_matrix:
        clf      = TSclassifier(clf=LDA())
        worker   = _one_perm_epoch
        acc_mean = float(acc_scores.mean())
    else:
        clf = (Pipeline([("scaler", StandardScaler()), ("lda", LDA(solver="svd"))])
               if args.normalize else LDA(solver="svd"))
        worker   = _one_perm_epoch_vector
        acc_mean = acc_scores.mean(axis=0)
    cv = StratifiedLeave2GroupsOut()
    # Instancie le même pipeline neuroscientifique que classify.py (TS+LDA ou LDA seul) selon le type de feature.

    # Checkpoints propres au fichier dst (jamais ceux du schéma subject, chemin différent).
    _clear_checkpoints(dst)

    perm = _run_perms_parallel(
        clf, cv, data, labels, n_trials, args.n_perm,
        args.key, args.state, args.n_jobs, args.checkpoint_every, dst, worker_fn=worker
    )
    # Distribution nulle avec permutation niveau epoch (Arthur), sur les MÊMES bootstraps que le schéma subject.

    result = dict(old)   # copie TOUT le contenu du résultat subject (acc_scores, n_subjects, ch_names, normalized...)
    result["perm_scheme"] = "epoch"
    result["perm_accs"]   = perm
    if is_matrix:
        result["pval"] = float((np.sum(perm >= acc_mean) + 1) / (args.n_perm + 1))
    else:
        result["pvals"] = (np.sum(perm >= acc_mean[None, :], axis=0) + 1) / (args.n_perm + 1)
    # Même formule de p-value que classify.py : (count + 1)/(n_perm + 1), Phipson & Smyth 2010.

    _save(dst, **result)
    _clear_checkpoints(dst)

    if is_matrix:
        print(f"pval (schéma epoch)   = {result['pval']:.4f}")
        print(f"pval (schéma subject, {src.name}) = {float(old['pval']):.4f}  <- pour comparaison côte à côte")
    else:
        print(f"pvals (schéma epoch)   min = {result['pvals'].min():.4f}")
        print(f"pvals (schéma subject, {src.name}) min = {np.array(old['pvals']).min():.4f}  <- pour comparaison côte à côte")
    print(f"Écrit : {dst}")
