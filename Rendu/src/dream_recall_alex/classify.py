"""Classification HR vs LR sur les features de sommeil.

Remplace classif_cov.py, classif_cosp.py et classif_psd.py du repo d'Arthur.
Lit les .npz atomiques produits par feat_extract, écrit un .npz de résultats par
couple (feature, stade) sous save-path/results/, plus un CSV de synthèse.

Deux modes selon la géométrie de la feature (cf config.MATRIX_KEYS) : les
matrices passent par TSclassifier(LDA()) en espace tangent riemannien, les
vecteurs par un LDA euclidien, une électrode à la fois.

n_trials_min est calculé une fois avant les jobs, sur la feature de référence,
ce qui garantit la comparabilité entre tous les états et toutes les features.

Le schéma de permutation, la validation croisée et l'absence de standardisation
sont justifiés dans le README (section Choix méthodologiques).

Reproductibilité : _seed() via hashlib.md5, déterministe entre exécutions,
contrairement à hash() dont le résultat dépend de PYTHONHASHSEED.

Checkpoint progressif (--checkpoint-every N) : sauvegarde tous les N bootstraps,
permet la reprise après timeout SLURM.
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
from pyriemann.classification import TSClassifier as TSclassifier

from .config import (
    CH_NAMES,
    CLASSIFICATION_GROUPS,
    FEATURE_KEYS,
    MATRIX_KEYS,
    N_EEG,
    PERM_SEED_OFFSET,
    REF_KEY,
    STATE_LIST,
    SUBJECT_LABELS,
    SUBJECT_LIST_ORDERED,
)
from .utils import load_atomic

_STATES_BY_LEN = sorted(STATE_LIST, key=len, reverse=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path",         type=Path, required=True)
    p.add_argument("--n-jobs",            type=int,  default=1)
    p.add_argument("--n-perm",            type=int,  default=1000)
    p.add_argument("--n-bootstraps",      type=int,  default=1000)
    p.add_argument("--checkpoint-every",  type=int,  default=50,
                   help="Sauvegarde checkpoint tous les N bootstraps (0=désactivé).")
    p.add_argument("--key",               type=str,  default=None,
                   help="Feature unique à classifier (ex: cov, cosp_sigma). "
                        "Si absent, toutes les features sont classifiées.")
    p.add_argument("--state",             type=str,  default=None,
                   help="Stade unique (ex: S2, SWS, NREM, REM). "
                        "Si absent, tous les stades sont classifiés.")
    p.add_argument("--skip-check",        action="store_true", default=False)
    p.add_argument("--overwrite",         action="store_true", default=False)
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────

def is_matrix_feature(key: str) -> bool:
    """Vrai si `key` désigne une feature matricielle (matrice SPD).

    L'appartenance est déclarée dans config.MATRIX_KEYS plutôt que déduite du
    nom : un nom commençant par `cosp_` ne garantit pas que la feature soit
    une matrice.
    """
    return key in MATRIX_KEYS

def _seed(key: str, state: str, idx: int) -> int:
    """Graine déterministe dérivée d'un hachage MD5 des `parts`.

    hash() de Python n'est pas déterministe entre deux exécutions (dépend de
    PYTHONHASHSEED), ce qui casserait la reproductibilité des tirages.
    """
    h = md5(f"{key}_{state}_{idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")

def load_subject(save_path: Path, key: str, sub_id: str, state: str) -> np.ndarray | None:
    """Charge et concatène les stades atomiques composant `state` pour un sujet.

    Les stades composites (SWS = S3+S4, NREM = S2+S3+S4) sont reconstruits ici
    à partir des .npz atomiques écrits par feat_extract.
    """
    stages = CLASSIFICATION_GROUPS[state]
    parts  = [a for s in stages if (a := load_atomic(save_path, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None

def load_all(save_path: Path, key: str, state: str) -> tuple[list, np.ndarray]:
    """Charge la feature `key` au stade `state` pour tous les sujets.

    Retourne (data, labels) : une liste d'arrays par sujet et le vecteur des
    labels HR/LR correspondants. Les sujets sans données sont ignorés.
    """
    data, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        arr = load_subject(save_path, key, sub_id, state)
        if arr is not None:
            data.append(arr)
            labels.append(label)
    return data, np.array(labels)

# ─── intégrité + n_trials_min ─────────────────────────────────────────────────

def compute_global_n_trials(save_path: Path, skip_check: bool = False) -> int:
    """Nombre d'epochs à tirer par sujet, commun à toutes les features et stades.

    Compte les epochs disponibles par sujet et par stade sur la feature de
    référence (REF_KEY = 'cov'), et retourne le minimum global. Ce minimum est
    le plus grand nombre d'epochs que l'on puisse tirer chez tous les sujets
    sans remise, d'où le nom n_trials_min, qui est en pratique un maximum
    utilisable.

    skip_check=False vérifie au passage que toutes les features ont le même
    nombre d'epochs que REF_KEY pour chaque sujet/stade.
    """
    ref_counts: dict[tuple[str, str], int] = {}
    for state in STATE_LIST:
        for sub_id in SUBJECT_LIST_ORDERED:
            arr = load_subject(save_path, REF_KEY, sub_id, state)
            if arr is not None:
                ref_counts[(sub_id, state)] = len(arr)

    if not ref_counts:
        raise RuntimeError(f"Aucun .npz '{REF_KEY}' trouvé, feat_extract complet ?")

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
    """Tire n_trials epochs par sujet, sans remise, et concatène.

    Équilibre la contribution des sujets : sans ce sous-tirage, un sujet à 300
    epochs pèserait deux fois plus qu'un sujet à 150. Retourne (X, y, groups)
    alignés par epoch, prêts pour la CV groupée.
    """
    rng = np.random.RandomState(seed)
    Xs, ys, gs = [], [], []
    for g, (arr, lab) in enumerate(zip(data, labels)):
        if len(arr) < n_trials:
            raise RuntimeError(
                f"Sujet groupe {g} : {len(arr)} epochs < n_trials={n_trials}."
            )
        idx = rng.choice(len(arr), size=n_trials, replace=False)
        Xs.append(arr[idx])
        ys.extend([lab] * n_trials)
        gs.extend([g]   * n_trials)
    return np.concatenate(Xs), np.array(ys), np.array(gs)

def permute_subject_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """Permute les labels HR/LR AU NIVEAU SUJET. Réf : Combrisson & Jerbi 2015."""
    return np.random.RandomState(seed).permutation(labels)

# Permutation NIVEAU EPOCH, réplique EXACTEMENT utils.py:103
# du repo arthurdehgan/sleep (fonction permutation_test). Utilisée uniquement
# par recompute_perms_epoch_arthur.py (script séparé), jamais par le pipeline
# principal ci-dessous (classify_matrix/classify_vector/main restent intacts).
def permute_epoch_labels(
    y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Permute les labels HR/LR AU NIVEAU EPOCH, réplique utils.py:103 d'Arthur.

    Arthur (permutation_test, github.com/arthurdehgan/sleep, utils.py) :
        perm_index = permutation(len(y))
        y_perm      = y[perm_index]
        groups_perm = groups[perm_index]
    Le MÊME index de permutation est appliqué à y ET à groups : les epochs
    sont mélangées globalement, labels et groupes réassignés ensemble. Chaque
    "groupe" (sujet) permuté devient un paquet aléatoire d'epochs des deux
    classes -> distribution nulle très resserrée -> p-values basses (mécanisme
    du p<0.001 de la thèse). Appliqué APRÈS le bootstrap (sur les tableaux
    concaténés y et groups produits par bootstrap_sample), contrairement au
    schéma subject qui permute AVANT (sur les labels sujets).
    """
    perm_index = np.random.RandomState(seed).permutation(len(y))
    return y[perm_index], groups[perm_index]


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
    # Isole les indices des époques de chaque classe (HR/LR) pour préparer un découpage stratifié.
    # Génère toutes les exclusions de groupes possibles (1 sujet par classe) grâce au produit cartésien.
    # Concatène et distribue à la volée les indices des sujets pour les blocs d'entraînement et de test.

    def get_n_splits(self, X, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        n = 1
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            n *= LeavePGroupsOut(1).get_n_splits(None, y[idx], groups[idx])
        return n
    # Calcule à l'avance le nombre total de splits (combinaisons) en multipliant le nombre de sujets HR par le nombre de sujets LR.

def run_cv(clf, splits, X, y) -> float:
    """Accuracy moyenne du classifieur sur tous les splits de la CV."""
    return float(np.mean([
        accuracy_score(y[te], clone(clf).fit(X[tr], y[tr]).predict(X[te]))
        for tr, te in splits
    ]))

# ─── bootstrap parallèle (1 bootstrap = 1 job) ────────────────────────────────

def _one_bootstrap(clf, cv, data, labels, n_trials, key, state, i) -> float:
    """Un seul bootstrap, appelé en parallèle par joblib.

    Échantillonne n_trials epochs par sujet (graine déterministe propre à
    l'itération), fige les 324 splits de CV (LPGO P=2) correspondant à ce
    tirage, et retourne l'accuracy moyenne sur ces splits.
    """
    X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)

def _one_perm(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> float:
    """Une seule permutation, appelée en parallèle par joblib.

    Permute les labels HR/LR au niveau sujet, échantillonne les epochs avec
    ces faux labels (graine décalée de PERM_SEED_OFFSET pour ne pas
    collisionner avec celles des bootstraps), et retourne le score nul.
    """
    labels_perm = permute_subject_labels(
        labels, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p)
    )
    X, y, groups = bootstrap_sample(
        data, labels_perm, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p)
    )
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)

# Equivalents vectoriels de _one_bootstrap/_one_perm ci-dessus, boucle
# sur les n_elec électrodes en interne, retourne un vecteur au lieu d'un float.
# Permettent à classify_vector de réutiliser _run_bootstraps_parallel /
# _run_perms_parallel (parallélisme + checkpoint) au lieu de sa double boucle
# Python séquentielle d'origine.

def _one_bootstrap_vector(clf, cv, data, labels, n_trials, key, state, i) -> np.ndarray:
    """Un seul bootstrap, appelé en parallèle par joblib (cas vectoriel).

    Même échantillonnage que _one_bootstrap, mais évalue chaque électrode
    séparément (1 LDA par colonne, sur les mêmes splits) et retourne le
    vecteur des n_elec scores.
    """
    X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
    splits = list(cv.split(X, y, groups))
    n_elec = X.shape[1]
    return np.array([
        run_cv(clf, splits, X[:, e:e + 1], y) for e in range(n_elec)
    ])

def _one_perm_vector(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> np.ndarray:
    """Une seule permutation, appelée en parallèle par joblib (cas vectoriel).

    Même schéma que _one_perm (permutation des labels au niveau sujet), mais
    retourne le vecteur des n_elec scores nuls — un par électrode, ce qui
    alimente la correction max-stat.
    """
    labels_perm = permute_subject_labels(
        labels, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p)
    )
    X, y, groups = bootstrap_sample(
        data, labels_perm, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p)
    )
    splits = list(cv.split(X, y, groups))
    n_elec = X.shape[1]
    return np.array([
        run_cv(clf, splits, X[:, e:e + 1], y) for e in range(n_elec)
    ])

# Workers de permutation NIVEAU EPOCH (schéma Arthur). Utilisés
# uniquement par recompute_perms_epoch_arthur.py (script séparé), le bootstrap
# est fait avec les VRAIS labels (identique aux bootstraps déjà calculés par le
# schéma subject), PUIS y et groups sont permutés ensemble au niveau epoch via
# permute_epoch_labels (utils.py:103 d'Arthur). Même signature que _one_perm/
# _one_perm_vector -> réutilisables tels quels par _run_perms_parallel via
# worker_fn, sans toucher à la machinerie de checkpoint existante.

def _one_perm_epoch(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> float:
    """Une permutation niveau epoch (matrice), réplique Arthur (utils.py:103)."""
    X, y, groups = bootstrap_sample(
        data, labels, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p)
    )
    y, groups = permute_epoch_labels(
        y, groups, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p)
    )
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)

def _one_perm_epoch_vector(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> np.ndarray:
    """Une permutation niveau epoch (vecteur), réplique Arthur (utils.py:103)."""
    X, y, groups = bootstrap_sample(
        data, labels, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p)
    )
    y, groups = permute_epoch_labels(
        y, groups, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p)
    )
    splits = list(cv.split(X, y, groups))
    n_elec = X.shape[1]
    return np.array([
        run_cv(clf, splits, X[:, e:e + 1], y) for e in range(n_elec)
    ])

# ─── checkpoint helpers ───────────────────────────────────────────────────────

def _ckpt_path(result_path: Path, prefix: str) -> Path:
    """Chemin du checkpoint temporaire (prefix = bootstrap | perm)."""
    return result_path.parent / (result_path.stem + f"_{prefix}_ckpt.npz")

def _load_checkpoint(result_path: Path, prefix: str) -> np.ndarray | None:
    """Charge un checkpoint partiel (bootstraps ou perms déjà calculés)."""
    p = _ckpt_path(result_path, prefix)
    if p.exists():
        return np.load(p)["data"]
    return None

def _save_checkpoint(result_path: Path, prefix: str, data: np.ndarray) -> None:
    """Écrit un checkpoint compressé, en créant le dossier parent si besoin."""
    p = _ckpt_path(result_path, prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, data=data)

def _clear_checkpoints(result_path: Path) -> None:
    """Supprime les checkpoints bootstrap et perm après sauvegarde finale."""
    for prefix in ["bootstrap", "perm"]:
        p = _ckpt_path(result_path, prefix)
        if p.exists():
            p.unlink()

# ─── bootstrap + perm loops avec checkpoint ───────────────────────────────────

# Ajout du paramètre worker_fn.
# Permet à classify_vector de réutiliser cette fonction via worker_fn=_one_bootstrap_vector.
# Comportement inchangé pour classify_matrix (valeur par défaut).
def _run_bootstraps_parallel(
    clf, cv, data, labels, n_trials, n_bootstraps, key, state,
    n_jobs, checkpoint_every, result_path, worker_fn=_one_bootstrap, prefer="threads"
) -> np.ndarray:
    """n_bootstraps bootstraps parallèles avec sauvegarde progressive.

    Reprend depuis le checkpoint si disponible (reprise après timeout) :
    l'index de départ est déduit du nombre de bootstraps déjà calculés.
    Les itérations restantes sont découpées en blocs distribués sur n_jobs
    workers joblib ; le checkpoint est réécrit à la fin de chaque bloc.
    checkpoint_every=0 désactive le checkpoint.
    """
    # Reprise depuis checkpoint
    done = _load_checkpoint(result_path, "bootstrap")
    start = len(done) if done is not None else 0
    accs = list(done) if done is not None else []

    if start >= n_bootstraps:
        print(f"    bootstrap: checkpoint complet ({start}/{n_bootstraps}), skip")
        return np.array(accs)

    if start > 0:
        print(f"    bootstrap: reprise depuis checkpoint ({start}/{n_bootstraps})")

    remaining = list(range(start, n_bootstraps))

    if checkpoint_every > 0:
        # par blocs pour le checkpoint
        for chunk_start in range(0, len(remaining), checkpoint_every):
            chunk = remaining[chunk_start: chunk_start + checkpoint_every]
            new_accs = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(worker_fn)(clf, cv, data, labels, n_trials, key, state, i)
                for i in chunk
            )
            accs.extend(new_accs)
            _save_checkpoint(result_path, "bootstrap", np.array(accs))
            print(f"    bootstrap: {len(accs)}/{n_bootstraps}")
    else:
        new_accs = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(worker_fn)(clf, cv, data, labels, n_trials, key, state, i)
            for i in remaining
        )
        accs.extend(new_accs)

    return np.array(accs)

# Même ajout de worker_fn que _run_bootstraps_parallel ci-dessus.
def _run_perms_parallel(
    clf, cv, data, labels, n_trials, n_perm, key, state,
    n_jobs, checkpoint_every, result_path, worker_fn=_one_perm, prefer="threads"
) -> np.ndarray:
    """n_perm permutations parallèles avec sauvegarde progressive.

    Même mécanique que _run_bootstraps_parallel : reprise depuis checkpoint,
    découpage en blocs sur n_jobs workers, réécriture du checkpoint à chaque
    bloc pour survivre aux limites de temps SLURM.
    """
    done = _load_checkpoint(result_path, "perm")
    start = len(done) if done is not None else 0
    perms = list(done) if done is not None else []

    if start >= n_perm:
        return np.array(perms)

    if start > 0:
        print(f"    perm: reprise depuis checkpoint ({start}/{n_perm})")

    remaining = list(range(start, n_perm))

    if checkpoint_every > 0:
        for chunk_start in range(0, len(remaining), checkpoint_every):
            chunk = remaining[chunk_start: chunk_start + checkpoint_every]
            new_perms = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(worker_fn)(clf, cv, data, labels, n_trials, key, state, p, n_perm)
                for p in chunk
            )
            perms.extend(new_perms)
            _save_checkpoint(result_path, "perm", np.array(perms))
            print(f"    perm: {len(perms)}/{n_perm}")
    else:
        new_perms = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(worker_fn)(clf, cv, data, labels, n_trials, key, state, p, n_perm)
            for p in remaining
        )
        perms.extend(new_perms)

    return np.array(perms)

# ─── cache helpers ────────────────────────────────────────────────────────────

def _result_path(save_path: Path, key: str, state: str) -> Path:
    """Chemin du .npz de résultats finaux pour un couple (feature, stade)."""
    return save_path / "results" / f"{key}_{state}.npz"

def _save(path: Path, **arrays) -> None:
    """Écrit `arrays` dans un .npz compressé, en créant le dossier parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

# ─── classification ───────────────────────────────────────────────────────────

def classify_matrix(save_path, key, state, n_trials, n_bootstraps, n_perm,
                    overwrite, n_jobs=1, checkpoint_every=50):
    """Classification riemannienne d'une feature matricielle (TSclassifier + LDA).

    Bootstraps et permutations parallélisés avec checkpoint. Pas de correction
    max-stat : un seul score global par couple (feature, stade), contrairement
    à classify_vector qui produit un score par électrode.
    """
    out = _result_path(save_path, key, state)
    if out.exists() and not overwrite:
        return np.load(out, allow_pickle=True)
    if overwrite:
        _clear_checkpoints(out)

    data, labels = load_all(save_path, key, state)
    if len(data) < 4:
        warnings.warn(
            f"Skipping {key}_{state}: cohorte insuffisante "
            f"(n={len(data)} sujets disponibles, minimum requis = 4)."
        )
        return None

    clf = TSclassifier(clf=LDA())
    cv  = StratifiedLeave2GroupsOut()

    acc_scores = _run_bootstraps_parallel(
        clf, cv, data, labels, n_trials, n_bootstraps,
        key, state, n_jobs, checkpoint_every, out, prefer="processes"
    )

    result = dict(
        acc_mean   = float(acc_scores.mean()),
        acc_std    = float(acc_scores.std()),
        acc_scores = acc_scores,
        n_trials   = int(n_trials),
        n_subjects = int(len(data)),
    )
    if n_perm > 0:
        perm = _run_perms_parallel(
            clf, cv, data, labels, n_trials, n_perm,
            key, state, n_jobs, checkpoint_every, out, prefer="processes"
        )
        result["pval"]      = float((np.sum(perm >= result["acc_mean"]) + 1) / (n_perm + 1))
        result["perm_accs"] = perm
        # pas de correction maxstat ici : un seul classifieur global, contrairement a classify_vector (1 score par electrode)

    _save(out, **result)
    _clear_checkpoints(out)  # nettoie les checkpoints après sauvegarde finale
    return result


def classify_vector(save_path, key, state, n_trials, n_bootstraps, n_perm,
                    overwrite, n_jobs=1, checkpoint_every=50):
    """Classification LDA euclidienne d'une feature vectorielle, par électrode.

    Un LDA univarié par électrode, sur les mêmes splits. Produit un vecteur de
    n_elec accuracies et, si n_perm > 0, un vecteur de p-values par électrode.
    Même mécanique de bootstraps et permutations parallélisés que
    classify_matrix.
    """
    # checkpoint_every=50 par défaut
    out = _result_path(save_path, key, state)
    if out.exists() and not overwrite:
        return np.load(out, allow_pickle=True)
    if overwrite:
        _clear_checkpoints(out)

    data, labels = load_all(save_path, key, state)
    if len(data) < 4:
        warnings.warn(
            f"Skipping {key}_{state}: cohorte insuffisante "
            f"(n={len(data)} sujets disponibles, minimum requis = 4)."
        )
        return None

    n_elec = data[0].shape[1]
    clf    = LDA(solver="svd")
    cv     = StratifiedLeave2GroupsOut()

    # _run_bootstraps_parallel + worker_fn=_one_bootstrap_vector, même mécanisme que classify_matrix.
    acc_scores = _run_bootstraps_parallel(
        clf, cv, data, labels, n_trials, n_bootstraps,
        key, state, n_jobs, checkpoint_every, out, worker_fn=_one_bootstrap_vector, prefer="processes"
    )

    result = dict(
        acc_mean   = acc_scores.mean(axis=0),
        acc_std    = acc_scores.std(axis=0),
        acc_scores = acc_scores,
        n_trials   = int(n_trials),
        n_subjects = int(len(data)),
        ch_names   = np.array(CH_NAMES[:N_EEG]),
    )
    if n_perm > 0:
        perm_accs = _run_perms_parallel(
            clf, cv, data, labels, n_trials, n_perm,
            key, state, n_jobs, checkpoint_every, out, worker_fn=_one_perm_vector, prefer="processes"
        )
        result["pvals"] = (
            np.sum(perm_accs >= result["acc_mean"][None, :], axis=0) + 1
        ) / (n_perm + 1)
        result["perm_accs"] = perm_accs

    _save(out, **result)
    _clear_checkpoints(out)
    return result

# ─── dispatcher ───────────────────────────────────────────────────────────────

def classify_one(save_path, key, state, n_trials, n_bootstraps, n_perm,
                 overwrite, n_jobs=1, checkpoint_every=50):
    """Route (key, state) vers classify_matrix ou classify_vector.

    Les exceptions sont interceptées et le traceback loggé plutôt que propagé :
    un combo qui plante ne doit pas faire tomber le reste du batch SLURM.
    """
    print(f"  {key} x {state}")
    try:
        fn = classify_matrix if is_matrix_feature(key) else classify_vector
        return key, state, fn(
            save_path, key, state, n_trials, n_bootstraps, n_perm,
            overwrite, n_jobs, checkpoint_every
        )
    except Exception:
        print(f"  ERROR {key} {state}\n{traceback.format_exc()}")
        return key, state, None


# ─── résumé CSV ───────────────────────────────────────────────────────────────

def build_summary_csv(save_path: Path) -> None:
    """Agrège les .npz de results/ en un CSV unique.

    Les features matricielles donnent une ligne (electrode='all'), les
    vectorielles une ligne par électrode. Les checkpoints sont ignorés.
    """
    rows, results_dir = [], save_path / "results"
    if not results_dir.exists():
        return

    for npz in sorted(results_dir.glob("*.npz")):
        # ignorer les fichiers checkpoint
        if "_ckpt" in npz.stem:
            continue
        stem  = npz.stem
        state = next((s for s in _STATES_BY_LEN if stem.endswith(f"_{s}")), None)
        if state is None:
            continue
        key        = stem[: -(len(state) + 1)]
        d          = np.load(npz, allow_pickle=True)
        acc_mean   = d["acc_mean"]
        acc_std    = d["acc_std"]
        n_trials   = int(d["n_trials"])
        pval_scalar = float(d["pval"]) if "pval" in d else np.nan

        if acc_mean.ndim == 0:
            rows.append(dict(key=key, state=state, electrode="all",
                             acc_mean=float(acc_mean), acc_std=float(acc_std),
                             n_trials=n_trials,
                             pval=pval_scalar,
                             pval_maxstat=float(d["pval_maxstat"]) if "pval_maxstat" in d else np.nan))
        else:
            ch_names      = d["ch_names"].tolist() if "ch_names" in d else list(range(len(acc_mean)))
            pvals         = d["pvals"]          if "pvals"         in d else [np.nan] * len(acc_mean)
            pvals_maxstat = d["pvals_maxstat"]  if "pvals_maxstat" in d else [np.nan] * len(acc_mean)
            for ch, am, astd, pv, pv_ms in zip(ch_names, acc_mean, acc_std, pvals, pvals_maxstat):
                rows.append(dict(key=key, state=state, electrode=ch,
                                 acc_mean=float(am), acc_std=float(astd),
                                 n_trials=n_trials,
                                 pval=float(pv), pval_maxstat=float(pv_ms)))

    if rows:
        out = results_dir / "classification_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"CSV : {out}")

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time()

    print("=== vérification intégrité + n_trials_min global ===")
    n_trials = compute_global_n_trials(args.save_path, skip_check=args.skip_check)
    print(f"n_trials_min = {n_trials}")

    # Filtrage par --key et --state si fournis (mode combo unique pour array SLURM)
    keys   = [args.key]   if args.key   else FEATURE_KEYS
    states = [args.state] if args.state else STATE_LIST
    combos = list(product(keys, states))
    print(f"=== classification : {len(combos)} combinaisons ===")

    # Un seul chemin :
    # classify_vector parallélise + checkpointe en interne comme les matrices,
    # cohérent aussi avec la topologie CCD/NUMA de Fir (1 combo à la fois sur
    # n_jobs cœurs, plutôt que dispersé entre combos).
    results = []
    for key, state in combos:
        res = classify_one(
            args.save_path, key, state, n_trials,
            args.n_bootstraps, args.n_perm, args.overwrite,
            n_jobs=args.n_jobs, checkpoint_every=args.checkpoint_every
        )
        results.append(res)

    print("\n=== résumé (features matricielles) ===")
    for key, state, res in sorted(results, key=lambda r: (r[1], r[0])):
        if res is not None and is_matrix_feature(key):
            print(f"  {key:20s} × {state:6s} : {float(res['acc_mean'])*100:.2f}%")

    build_summary_csv(args.save_path)
    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal : {m}m{s:02d}s")


if __name__ == "__main__":
    main()
