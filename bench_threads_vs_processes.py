"""Benchmark direct threads vs processes pour une feature MATRICIELLE (cov),
a 32 coeurs, sur le meme combo. Mesure ce qui n'avait ete teste que pour les
vecteurs dans "Opti 64c" -- on verifie empiriquement si le raisonnement
"BLAS libere le GIL -> threads suffit pour les matrices" tient en pratique.

Lance N_BENCH bootstraps en threads, puis N_BENCH en processes, chronometre
les deux. Aucune ecriture de resultat -- juste une mesure de temps.

Usage:
    python bench_threads_vs_processes.py --n-jobs 32 --n-bench 96
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.classification import TSClassifier as TSclassifier

sys.path.insert(0, "/home/alouis/dream-recall-alex")
from classify import (
    load_all, bootstrap_sample, run_cv, _seed,
    StratifiedLeave2GroupsOut,
)

SAVE_PATH = Path("/home/alouis/scratch/dream_features_baseline52")
KEY, STATE, N_TRIALS = "cov", "S2", 52


def one_bootstrap(clf, cv, data, labels, n_trials, key, state, i):
    X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--n-bench", type=int, default=96)
    p.add_argument("--backend", type=str, default=None, choices=["threads", "processes"],
                   help="Si fourni, ne teste QUE ce backend (permet de lancer threads et "
                        "processes en 2 jobs SLURM separes, en parallele, au lieu de "
                        "sequentiellement dans le meme job).")
    args = p.parse_args()

    data, labels = load_all(SAVE_PATH, KEY, STATE)
    clf = TSclassifier(clf=LDA())
    cv = StratifiedLeave2GroupsOut()
    # indices 600-699 : au-dela du checkpoint existant (550), evite de retomber
    # sur le bootstrap 551 qui plantait (positive definite) -- on veut chronometrer,
    # pas crasher. Ces indices sont juste pour le timing, pas sauvegardes.
    idx = list(range(600, 600 + args.n_bench))

    print(f"Benchmark {args.n_bench} bootstraps, n_jobs={args.n_jobs}, feature={KEY}/{STATE}", flush=True)

    backends = [args.backend] if args.backend else ["threads", "processes"]
    for backend in backends:
        t0 = time.time()
        _ = Parallel(n_jobs=args.n_jobs, prefer=backend)(
            delayed(one_bootstrap)(clf, cv, data, labels, N_TRIALS, KEY, STATE, i)
            for i in idx
        )
        dt = time.time() - t0
        print(f"  prefer={backend:10s} : {dt:6.1f}s total  ({dt/args.n_bench*1000:.0f} ms/bootstrap)", flush=True)


if __name__ == "__main__":
    main()
