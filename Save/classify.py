"""Classification HR vs LR sur les features de sommeil.

Remplace classif_cov.py, classif_cosp.py, classif_psd.py du repo Arthur.
Lit les .npz atomiques produits par feat_extract.py.

Deux modes selon le type de feature :
- Matrice (cov, cosp_*) : TSclassifier(LDA()) en espace Riemannien,
  StratifiedLeave2GroupsOut (LPGO P=2 stratifié HR/LR, §1.2.7 thèse).
- Vecteur (psd_*, psd_osc_*, aperiodic) : LDA Euclidien par électrode.
  --normalize active StandardScaler fit sur train uniquement (off par
  défaut pour rester cohérent avec Arthur).

n_trials_min global calculé depuis 'cov' avant les jobs (comparabilité
garantie entre tous les états/features). Vérification d'intégrité optionnelle.

Reproductibilité : _seed() via hashlib.md5 (déterministe cross-platform,
contrairement à hash() Python dont le résultat dépend de PYTHONHASHSEED).

Usage :
    python classify.py \\
        --save-path /path/to/dream_features \\
        --n-jobs    $SLURM_CPUS_PER_TASK \\
        --n-perm    0 \\
        --normalize \\
        --skip-check
"""

import argparse
import traceback
import warnings
from hashlib import md5
from itertools import product
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeavePGroupsOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.classification import TSclassifier

from config import (
    CH_NAMES,
    CLASSIFICATION_GROUPS,
    FEATURE_KEYS,
    N_EEG,
    STATE_LIST,
    SUBJECT_LABELS,
    SUBJECT_LIST_ORDERED,
)
from utils import load_atomic

PERM_SEED_OFFSET = 100_003  # >> n_bootstraps : garantit que les seeds perms
                            # n'entrent pas en collision avec les seeds bootstrap
REF_KEY          = "cov"
# STATE_LIST trié par longueur décroissante pour éviter les faux positifs
# dans build_summary_csv (ex: "REM" sous-chaîne de "NREM")
_STATES_BY_LEN   = sorted(STATE_LIST, key=len, reverse=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path",    type=Path, required=True)
    p.add_argument("--n-jobs",       type=int,  default=1)
    p.add_argument("--n-perm",       type=int,  default=0)
    p.add_argument("--n-bootstraps", type=int,  default=1000)
    p.add_argument("--normalize",    action="store_true", default=False,
                   help="StandardScaler sur train (vecteur uniquement, off=cohérent Arthur).")
    p.add_argument("--skip-check",   action="store_true", default=False,
                   help="Skip la vérification d'intégrité des .npz.")
    p.add_argument("--overwrite",    action="store_true", default=False)
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────

def is_matrix_feature(key: str) -> bool:
    return key == "cov" or key.startswith("cosp_")


def _seed(key: str, state: str, idx: int) -> int:
    """Hash déterministe cross-platform (md5, pas hash() Python)."""
    h = md5(f"{key}_{state}_{idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def load_subject(save_path: Path, key: str, sub_id: str, state: str) -> np.ndarray | None:
    stages = CLASSIFICATION_GROUPS[state]
    parts  = [a for s in stages if (a := load_atomic(save_path, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None


def load_all(save_path: Path, key: str, state: str) -> tuple[list, np.ndarray]:
    data, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        arr = load_subject(save_path, key, sub_id, state)
        if arr is not None:
            data.append(arr)
            labels.append(label)
    return data, np.array(labels)


# ─── intégrité + n_trials_min ─────────────────────────────────────────────────

def compute_global_n_trials(save_path: Path, skip_check: bool = False) -> int:
    """n_trials_min depuis REF_KEY uniquement — propriété du signal, pas de la feature."""
    ref_counts: dict[tuple[str, str], int] = {}
    for state in STATE_LIST:
        for sub_id in SUBJECT_LIST_ORDERED:
            arr = load_subject(save_path, REF_KEY, sub_id, state)
            if arr is not None:
                ref_counts[(sub_id, state)] = len(arr)

    if not ref_counts:
        raise RuntimeError(f"Aucun .npz '{REF_KEY}' trouvé — feat_extract complet ?")

    if not skip_check:
        missing = []
        for key in FEATURE_KEYS:
            if key == REF_KEY:
                continue
            for (sub_id, state), n_ref in ref_counts.items():
                arr = load_subject(save_path, key, sub_id, state)
                if arr is None:
                    missing.append(f"{key} / sub-{sub_id} / {state} : absent")
                elif len(arr) != n_ref:
                    missing.append(
                        f"{key} / sub-{sub_id} / {state} : "
                        f"{len(arr)} epochs vs {n_ref} dans {REF_KEY}"
                    )
        if missing:
            raise RuntimeError(
                f"feat_extract incomplet ou incohérent ({len(missing)} cas) :\n"
                + "\n".join(missing[:20])
                + ("\n  ..." if len(missing) > 20 else "")
            )

    return int(min(ref_counts.values()))


# ─── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_sample(
    data: list, labels: np.ndarray, n_trials: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    Xs, ys, gs = [], [], []
    for g, (arr, lab) in enumerate(zip(data, labels)):
        if len(arr) < n_trials:
            raise RuntimeError(
                f"Sujet groupe {g} : {len(arr)} epochs < n_trials={n_trials}. "
                "Relancer compute_global_n_trials ou vérifier feat_extract."
            )
        idx = rng.choice(len(arr), size=n_trials, replace=False)
        Xs.append(arr[idx])
        ys.extend([lab] * n_trials)
        gs.extend([g]   * n_trials)
    return np.concatenate(Xs), np.array(ys), np.array(gs)


def permute_subject_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """Permute les labels HR/LR AU NIVEAU SUJET (pas epoch).

    Le label est une propriété du sujet : permuter les epochs casserait la
    cohérence intra-sujet et gonflerait la distribution nulle (p-value trop
    optimiste). Réf : Combrisson & Jerbi 2015, cité dans la thèse §1.2.7.
    """
    return np.random.RandomState(seed).permutation(labels)


# ─── cross-validation ─────────────────────────────────────────────────────────

class StratifiedLeave2GroupsOut:
    """LPGO P=2 stratifié : 1 sujet HR + 1 sujet LR en test à chaque split."""

    def split(self, X, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Attendu 2 classes, obtenu {len(classes)}")
        idx_per_cls = [np.where(y == c)[0] for c in classes]
        iters = [
            list(LeavePGroupsOut(1).split(np.arange(len(idx)), y[idx], groups[idx]))
            for idx in idx_per_cls
        ]
        for s0, s1 in product(iters[0], iters[1]):
            yield (
                np.concatenate([idx_per_cls[0][s0[0]], idx_per_cls[1][s1[0]]]),
                np.concatenate([idx_per_cls[0][s0[1]], idx_per_cls[1][s1[1]]]),
            )

    def get_n_splits(self, X, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        n = 1
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            n *= LeavePGroupsOut(1).get_n_splits(None, y[idx], groups[idx])
        return n


def run_cv(clf, splits, X, y) -> float:
    """Accuracy moyenne sur des splits précalculés (réutilisables entre électrodes)."""
    return float(np.mean([
        accuracy_score(y[te], clone(clf).fit(X[tr], y[tr]).predict(X[te]))
        for tr, te in splits
    ]))


# ─── bootstrap + perm loops ───────────────────────────────────────────────────

def _run_bootstraps(clf, cv, data, labels, n_trials, n_bootstraps, key, state) -> np.ndarray:
    accs = []
    for i in range(n_bootstraps):
        X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
        splits = list(cv.split(X, y, groups))
        accs.append(run_cv(clf, splits, X, y))
    return np.array(accs)


def _run_perms(clf, cv, data, labels, n_trials, n_perm, key, state) -> np.ndarray:
    out = []
    for p in range(n_perm):
        labels_perm = permute_subject_labels(labels, _seed(key, state, PERM_SEED_OFFSET + n_perm + p))
        X, y, groups = bootstrap_sample(data, labels_perm, n_trials, _seed(key, state, PERM_SEED_OFFSET + p))
        splits = list(cv.split(X, y, groups))
        out.append(run_cv(clf, splits, X, y))
    return np.array(out)


# ─── cache helpers ────────────────────────────────────────────────────────────

def _result_path(save_path: Path, key: str, state: str) -> Path:
    return save_path / "results" / f"{key}_{state}.npz"


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ─── classification ───────────────────────────────────────────────────────────

def classify_matrix(save_path, key, state, n_trials, n_bootstraps, n_perm, overwrite, normalize):
    if normalize:
        warnings.warn(f"--normalize ignoré pour la feature matricielle '{key}'.")

    out = _result_path(save_path, key, state)
    if out.exists() and not overwrite:
        return np.load(out, allow_pickle=True)

    data, labels = load_all(save_path, key, state)
    if len(data) < 4:
        return None

    clf, cv    = TSclassifier(clf=LDA()), StratifiedLeave2GroupsOut()
    acc_scores = _run_bootstraps(clf, cv, data, labels, n_trials, n_bootstraps, key, state)

    result = dict(
        acc_mean   = float(acc_scores.mean()),
        acc_std    = float(acc_scores.std()),
        acc_scores = acc_scores,
        n_trials   = int(n_trials),
        n_subjects = int(len(data)),
        normalized = False,
    )
    if n_perm > 0:
        perm = _run_perms(clf, cv, data, labels, n_trials, n_perm, key, state)
        result["pval"]      = float((np.sum(perm >= result["acc_mean"]) + 1) / (n_perm + 1))
        result["perm_accs"] = perm

    _save(out, **result)
    return result


def classify_vector(save_path, key, state, n_trials, n_bootstraps, n_perm, overwrite, normalize):
    out = _result_path(save_path, key, state)
    if out.exists() and not overwrite:
        return np.load(out, allow_pickle=True)

    data, labels = load_all(save_path, key, state)
    if len(data) < 4:
        return None

    n_elec     = data[0].shape[1]
    clf        = (Pipeline([("scaler", StandardScaler()), ("lda", LDA(solver="svd"))])
                  if normalize else LDA(solver="svd"))
    cv         = StratifiedLeave2GroupsOut()
    acc_scores = np.zeros((n_bootstraps, n_elec))

    for i in range(n_bootstraps):
        X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
        splits = list(cv.split(X, y, groups))  # identiques pour les 19 électrodes
        for e in range(n_elec):
            acc_scores[i, e] = run_cv(clf, splits, X[:, e:e+1], y)

    result = dict(
        acc_mean   = acc_scores.mean(axis=0),
        acc_std    = acc_scores.std(axis=0),
        acc_scores = acc_scores,
        n_trials   = int(n_trials),
        n_subjects = int(len(data)),
        ch_names   = np.array(CH_NAMES[:N_EEG]),
        normalized = normalize,
    )
    if n_perm > 0:
        perm_accs = np.zeros((n_perm, n_elec))
        for p in range(n_perm):
            labels_perm = permute_subject_labels(labels, _seed(key, state, PERM_SEED_OFFSET + n_perm + p))
            X, y, groups = bootstrap_sample(data, labels_perm, n_trials, _seed(key, state, PERM_SEED_OFFSET + p))
            splits = list(cv.split(X, y, groups))
            for e in range(n_elec):
                perm_accs[p, e] = run_cv(clf, splits, X[:, e:e+1], y)
        result["pvals"]      = (np.sum(perm_accs >= result["acc_mean"][None, :], axis=0) + 1) / (n_perm + 1)
        result["perm_accs"]  = perm_accs

    _save(out, **result)
    return result


# ─── dispatcher ───────────────────────────────────────────────────────────────

def classify_one(save_path, key, state, n_trials, n_bootstraps, n_perm, overwrite, normalize):
    print(f"  {key} × {state}")
    try:
        fn = classify_matrix if is_matrix_feature(key) else classify_vector
        return key, state, fn(save_path, key, state, n_trials, n_bootstraps, n_perm, overwrite, normalize)
    except Exception:
        print(f"  ERROR {key} {state}\n{traceback.format_exc()}")
        return key, state, None


# ─── résumé CSV ───────────────────────────────────────────────────────────────

def build_summary_csv(save_path: Path) -> None:
    rows, results_dir = [], save_path / "results"
    if not results_dir.exists():
        return

    for npz in sorted(results_dir.glob("*.npz")):
        stem  = npz.stem
        state = next((s for s in _STATES_BY_LEN if stem.endswith(f"_{s}")), None)
        if state is None:
            continue
        key        = stem[: -(len(state) + 1)]
        d          = np.load(npz, allow_pickle=True)
        acc_mean   = d["acc_mean"]
        acc_std    = d["acc_std"]
        n_trials   = int(d["n_trials"])
        normalized = bool(d["normalized"]) if "normalized" in d else False
        pval_scalar = float(d["pval"]) if "pval" in d else np.nan

        if acc_mean.ndim == 0:
            rows.append(dict(key=key, state=state, electrode="all",
                             acc_mean=float(acc_mean), acc_std=float(acc_std),
                             n_trials=n_trials, normalized=normalized, pval=pval_scalar))
        else:
            ch_names = d["ch_names"].tolist() if "ch_names" in d else list(range(len(acc_mean)))
            pvals    = d["pvals"] if "pvals" in d else [np.nan] * len(acc_mean)
            for ch, am, astd, pv in zip(ch_names, acc_mean, acc_std, pvals):
                rows.append(dict(key=key, state=state, electrode=ch,
                                 acc_mean=float(am), acc_std=float(astd),
                                 n_trials=n_trials, normalized=normalized, pval=float(pv)))

    if rows:
        out = results_dir / "classification_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"CSV : {out}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    t0   = time()

    print("=== vérification intégrité + n_trials_min global ===")
    n_trials = compute_global_n_trials(args.save_path, skip_check=args.skip_check)
    print(f"n_trials_min = {n_trials}  |  normalize = {args.normalize}")

    combos = list(product(FEATURE_KEYS, STATE_LIST))
    print(f"=== classification : {len(combos)} combinaisons ===")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(classify_one)(
            args.save_path, key, state, n_trials,
            args.n_bootstraps, args.n_perm, args.overwrite, args.normalize,
        )
        for key, state in combos
    )

    print("\n=== résumé (features matricielles) ===")
    for key, state, res in sorted(results, key=lambda r: (r[1], r[0])):
        if res is not None and is_matrix_feature(key):
            print(f"  {key:20s} × {state:6s} : {float(res['acc_mean'])*100:.2f}%")

    build_summary_csv(args.save_path)
    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal : {m}m{s:02d}s")
