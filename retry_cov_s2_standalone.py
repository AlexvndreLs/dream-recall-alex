"""Complete cov/S2 (baseline52, n_trials=52) au-dela du checkpoint existant
(550/1000 bootstraps), avec retry-seed en cas de ValueError "positive definite",
SANS MODIFIER classify.py -- script autonome, reutilise ses fonctions par import
pour eviter toute divergence de logique.

Parallelise via joblib (n_jobs configurable), comme classify.py, pour tourner
en sbatch 18h sur 32 coeurs -- PAS en salloc mono-coeur (trop lent).

Ecrit dans un fichier SEPARE, aucune contamination du pipeline principal.

Usage:
    python retry_cov_s2_standalone.py --n-jobs 32
"""
import argparse
import sys
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
KEY, STATE = "cov", "S2"
N_TRIALS = 52
N_BOOTSTRAPS = 1000
MAX_RETRY = 5
CHECKPOINT_EVERY = 50

EXISTING_CKPT = SAVE_PATH / "results" / f"{KEY}_{STATE}_bootstrap_ckpt.npz"
OUT_FILE = SAVE_PATH / "results" / f"{KEY}_{STATE}_bootstrap_RETRY_STANDALONE.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=32)
    return p.parse_args()


def one_bootstrap_with_retry(clf, cv, data, labels, n_trials, key, state, i, max_retry=MAX_RETRY):
    """Identique a _one_bootstrap de classify.py + retry sur seed alternative en
    cas de ValueError 'positive definite'. HYPOTHESE NON VERIFIEE : un tirage
    different evite le meme conditionnement degenere -- pas garanti, teste ici."""
    for attempt in range(max_retry):
        seed = _seed(key, state, i) if attempt == 0 else _seed(key, state, i) + attempt * 1_000_000
        try:
            X, y, groups = bootstrap_sample(data, labels, n_trials, seed)
            splits = list(cv.split(X, y, groups))
            return run_cv(clf, splits, X, y)
        except ValueError as e:
            if "positive definite" not in str(e):
                raise
            print(f"  [retry {attempt+1}/{max_retry}] bootstrap {i} : seed={seed} degenere, nouvelle seed...", flush=True)
    raise RuntimeError(f"bootstrap {i} : echec apres {max_retry} tentatives (probleme structurel)")


def main():
    args = parse_args()

    if OUT_FILE.exists():
        accs = list(np.load(OUT_FILE)["data"])
        print(f"Repris {len(accs)} bootstraps depuis la progression precedente de CE script", flush=True)
    elif EXISTING_CKPT.exists():
        accs = list(np.load(EXISTING_CKPT)["data"])
        print(f"Repris {len(accs)} bootstraps valides depuis le checkpoint classify.py existant", flush=True)
    else:
        accs = []
        print("Aucun checkpoint existant, depart de zero", flush=True)

    data, labels = load_all(SAVE_PATH, KEY, STATE)
    clf = TSclassifier(clf=LDA())
    cv = StratifiedLeave2GroupsOut()

    remaining = list(range(len(accs), N_BOOTSTRAPS))
    # par blocs pour le checkpoint (comme classify.py)
    for chunk_start in range(0, len(remaining), CHECKPOINT_EVERY):
        chunk = remaining[chunk_start: chunk_start + CHECKPOINT_EVERY]
        # prefer="threads" (PAS "processes") : cov est une feature MATRICIELLE.
        # Mesure empirique (conv "Opti 64c") : TSclassifier fait de gros calculs
        # BLAS (numpy.linalg.eigh via gmean = 88% du temps) qui LIBERENT le GIL,
        # donc les threads parallelisent deja bien. processes serait plus lent
        # (cout de pickle de data vers chaque process, sans gain). Coherent avec
        # classify_matrix qui garde "threads" dans classify.py.
        new = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(one_bootstrap_with_retry)(clf, cv, data, labels, N_TRIALS, KEY, STATE, i)
            for i in chunk
        )
        accs.extend(new)
        np.savez_compressed(OUT_FILE, data=np.array(accs))
        print(f"  {len(accs)}/{N_BOOTSTRAPS} (checkpoint standalone)", flush=True)

    accs = np.array(accs)
    np.savez_compressed(OUT_FILE, data=accs)
    print(f"\n=== cov/S2 (baseline52, retry-standalone) ===", flush=True)
    print(f"n={len(accs)}  moyenne={accs.mean():.4f}  std={accs.std():.4f}", flush=True)
    print(f"Sortie : {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
