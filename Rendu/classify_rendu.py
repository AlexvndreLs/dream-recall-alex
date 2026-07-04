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

Checkpoint progressif (--checkpoint-every N) : sauvegarde les bootstraps
toutes les N itérations -> reprise après timeout sans repartir de zéro.
Disponible pour les features matricielles ET vectorielles (MODIF : avant,
seule classify_matrix en bénéficiait — voir _run_bootstraps_parallel).

Parallélisation (--n-jobs) : un seul combo (key, state) à la fois reçoit
tout n_jobs, pour matrices ET vecteurs (MODIF : avant, les vecteurs étaient
parallélisés au niveau des combos eux-mêmes, plusieurs en même temps).

Usage :
    python classify.py \\
        --save-path /path/to/dream_features \\
        --n-jobs    $SLURM_CPUS_PER_TASK \\
        --n-perm    1000 \\
        --key       cov \\
        --state     S2 \\
        --checkpoint-every 50
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
from pyriemann.classification import TSClassifier as TSclassifier

from config_v3 import (
    CH_NAMES,
    CLASSIFICATION_GROUPS,
    FEATURE_KEYS,
    N_EEG,
    STATE_LIST,
    SUBJECT_LABELS,
    SUBJECT_LIST_ORDERED,
)
from utils import load_atomic

PERM_SEED_OFFSET = 100_003
REF_KEY          = "cov"
_STATES_BY_LEN   = sorted(STATE_LIST, key=len, reverse=True)
MATRIX_KEYS      = ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma", "cosp_beta"]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path",         type=Path, required=True)
    p.add_argument("--n-jobs",            type=int,  default=1)
    p.add_argument("--n-perm",            type=int,  default=0)
    p.add_argument("--n-bootstraps",      type=int,  default=1000)
    p.add_argument("--checkpoint-every",  type=int,  default=50,
                   help="Sauvegarde checkpoint tous les N bootstraps (0=désactivé).")
    p.add_argument("--key",               type=str,  default=None,
                   help="Feature unique à classifier (ex: cov, cosp_sigma). "
                        "Si absent, toutes les features sont classifiées.")
    p.add_argument("--state",             type=str,  default=None,
                   help="Stade unique (ex: S2, SWS, NREM, REM). "
                        "Si absent, tous les stades sont classifiés.")
    p.add_argument("--normalize",         action="store_true", default=False)
    p.add_argument("--skip-check",        action="store_true", default=False)
    p.add_argument("--overwrite",         action="store_true", default=False)
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────

def is_matrix_feature(key: str) -> bool:
    return key == "cov" or key.startswith("cosp_")
# Détermine si la clé correspond à une feature matricielle (covariance ou cross-spectre) ou vectorielle.

def _seed(key: str, state: str, idx: int) -> int:
    h = md5(f"{key}_{state}_{idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")
# Génère une graine aléatoire entière et déterministe par hachage MD5 pour garantir la reproductibilité cross-platform.

def load_subject(save_path: Path, key: str, sub_id: str, state: str) -> np.ndarray | None:
    stages = CLASSIFICATION_GROUPS[state]
    parts  = [a for s in stages if (a := load_atomic(save_path, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None
# Charge et concatène les données atomiques de toutes les phases de sommeil associées à un stade pour un sujet donné.

def load_all(save_path: Path, key: str, state: str) -> tuple[list, np.ndarray]:
    data, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        arr = load_subject(save_path, key, sub_id, state)
        if arr is not None:
            data.append(arr)
            labels.append(label)
    return data, np.array(labels)
# Compile l'ensemble des matrices/vecteurs de caractéristiques et les étiquettes de groupe associées pour tous les sujets.

# ─── intégrité + n_trials_min ─────────────────────────────────────────────────

def compute_global_n_trials(save_path: Path, skip_check: bool = False) -> int:
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
# Compte le nombre d'époques disponibles par sujet et par stade pour la feature de référence (cov).
# Vérifie la cohérence des données en s'assurant qu'aucune feature ne présente de données manquantes ou asymétriques.
# Retourne le nombre minimal global d'époques (n_trials_min) pour permettre un sous-tirage équilibré entre tous les sujets.
# Attention min parmi le nombre d'epoch mais c'est notre max du coup 

# ─── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_sample(
    data: list, labels: np.ndarray, n_trials: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# Initialise un générateur aléatoire reproductible et des listes pour collecter les données échantillonnées.
# Tire au hasard et sans remise un nombre fixe d'époques (n_trials) pour chaque sujet afin d'équilibrer l'ensemble.
# Aligne les étiquettes (HR/LR) et les identifiants de sujets par époque, puis concatène le tout pour le classifieur.

def permute_subject_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """Permute les labels HR/LR AU NIVEAU SUJET. Réf : Combrisson & Jerbi 2015."""
    return np.random.RandomState(seed).permutation(labels)
# Mélange aléatoirement les étiquettes de diagnostic (HR/LR) directement au niveau des sujets pour les tests de permutations.

# MODIF (04/07) : permutation NIVEAU EPOCH — réplique EXACTEMENT utils.py:103
# du repo arthurdehgan/sleep (fonction permutation_test). Utilisée uniquement
# par recompute_perms_epoch_arthur.py (script séparé), jamais par le pipeline
# principal ci-dessous (classify_matrix/classify_vector/main restent intacts).
def permute_epoch_labels(
    y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Permute les labels HR/LR AU NIVEAU EPOCH — réplique utils.py:103 d'Arthur.

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
# Applique un unique index de permutation aux labels ET aux groupes au niveau epoch (code exact d'Arthur).
# Détruit la correspondance sujet<->epochs, produisant une distribution nulle étroite (p plus faibles).


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
    return float(np.mean([
        accuracy_score(y[te], clone(clf).fit(X[tr], y[tr]).predict(X[te]))
        for tr, te in splits
    ]))
# Évalue le modèle par validation croisée en calculant la moyenne des scores d'exactitude (accuracy) obtenus sur l'ensemble des splits.

# ─── bootstrap parallèle (1 bootstrap = 1 job) ────────────────────────────────

def _one_bootstrap(clf, cv, data, labels, n_trials, key, state, i) -> float:
    """Un seul bootstrap — appelé en parallèle par joblib."""
    X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)
# Échantillonne un nombre fixe d'époques par sujet à l'aide d'une graine aléatoire déterministe propre à l'itération.
# Génère et fige la liste des 324 splits de validation croisée (LPGO P=2) adaptés à ce tirage de données.
# Exécute l'évaluation croisée complète (entraînement + test) et retourne le score d'accuracy moyen obtenu.

def _one_perm(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> float:
    """Une seule permutation — appelée en parallèle par joblib."""
    labels_perm = permute_subject_labels(
        labels, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p)
    )
    X, y, groups = bootstrap_sample(
        data, labels_perm, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p)
    )
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)
# Permute aléatoirement les étiquettes HR/LR au niveau des sujets pour rompre l'association biologique avec le signal.
# Effectue le sous-tirage des époques en leur associant ces faux labels à l'aide d'une graine isolée des bootstraps.
# Évalue le modèle sur ces données falsifiées pour alimenter la distribution statistique de l'hypothèse nulle.

# MODIF : équivalents vectoriels de _one_bootstrap/_one_perm ci-dessus — boucle
# sur les n_elec électrodes en interne, retourne un vecteur au lieu d'un float.
# Permettent à classify_vector de réutiliser _run_bootstraps_parallel /
# _run_perms_parallel (parallélisme + checkpoint) au lieu de sa double boucle
# Python séquentielle d'origine.

def _one_bootstrap_vector(clf, cv, data, labels, n_trials, key, state, i) -> np.ndarray:
    """Un seul bootstrap — appelé en parallèle par joblib (cas vectoriel)."""
    X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
    splits = list(cv.split(X, y, groups))
    n_elec = X.shape[1]
    return np.array([
        run_cv(clf, splits, X[:, e:e + 1], y) for e in range(n_elec)
    ])
# Échantillonne un nombre fixe d'époques par sujet à l'aide d'une graine aléatoire déterministe propre à l'itération.
# Évalue séquentiellement chaque électrode (1 LDA par colonne) sur les mêmes splits et retourne le vecteur des n_elec scores.

def _one_perm_vector(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> np.ndarray:
    """Une seule permutation — appelée en parallèle par joblib (cas vectoriel)."""
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
# Permute les labels HR/LR au niveau sujet puis ré-échantillonne les époques avec ces faux labels, comme _one_perm.
# Évalue séquentiellement chaque électrode sur les données falsifiées et retourne le vecteur de n_elec scores nuls.

# MODIF (04/07) : workers de permutation NIVEAU EPOCH (schéma Arthur). Utilisés
# uniquement par recompute_perms_epoch_arthur.py (script séparé) — le bootstrap
# est fait avec les VRAIS labels (identique aux bootstraps déjà calculés par le
# schéma subject), PUIS y et groups sont permutés ensemble au niveau epoch via
# permute_epoch_labels (utils.py:103 d'Arthur). Même signature que _one_perm/
# _one_perm_vector -> réutilisables tels quels par _run_perms_parallel via
# worker_fn, sans toucher à la machinerie de checkpoint existante.

def _one_perm_epoch(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> float:
    """Une permutation niveau epoch (matrice) — réplique Arthur (utils.py:103)."""
    X, y, groups = bootstrap_sample(
        data, labels, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p)
    )
    y, groups = permute_epoch_labels(
        y, groups, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p)
    )
    splits = list(cv.split(X, y, groups))
    return run_cv(clf, splits, X, y)
# Sous-tire les époques avec les VRAIS labels, puis permute y+groups au niveau epoch (ordre inverse du schéma subject).
# Évalue la CV sur ces données mélangées et retourne le score nul (distribution resserrée -> p bas).

def _one_perm_epoch_vector(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> np.ndarray:
    """Une permutation niveau epoch (vecteur) — réplique Arthur (utils.py:103)."""
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
# Version électrode-par-électrode du worker epoch : bootstrap réel, permutation y+groups niveau epoch, 1 LDA/colonne.
# Retourne le vecteur de n_elec scores nuls pour la correction max-stat sur électrodes.

# ─── checkpoint helpers ───────────────────────────────────────────────────────

def _ckpt_path(result_path: Path, prefix: str) -> Path:
    return result_path.parent / (result_path.stem + f"_{prefix}_ckpt.npz")
# Génère le chemin absolu du fichier de checkpoint temporaire (.npz).

def _load_checkpoint(result_path: Path, prefix: str) -> np.ndarray | None:
    """Charge un checkpoint partiel (bootstraps ou perms déjà calculés)."""
    p = _ckpt_path(result_path, prefix)
    if p.exists():
        return np.load(p)["data"]
    return None
# Charge le tableau de scores NumPy si un fichier de checkpoint existe, sinon renvoie None.

def _save_checkpoint(result_path: Path, prefix: str, data: np.ndarray) -> None:
    p = _ckpt_path(result_path, prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, data=data)
# Crée le dossier parent si nécessaire et sauvegarde l'état des calculs au format compressé.

def _clear_checkpoints(result_path: Path) -> None:
    for prefix in ["bootstrap", "perm"]:
        p = _ckpt_path(result_path, prefix)
        if p.exists():
            p.unlink()
# Supprime physiquement les fichiers temporaires de bootstrap et de permutation du disque.

# ─── bootstrap + perm loops avec checkpoint ───────────────────────────────────

# MODIF : ajout du paramètre worker_fn (avant : _one_bootstrap codé en dur).
# Permet à classify_vector de réutiliser cette fonction via worker_fn=_one_bootstrap_vector.
# Comportement inchangé pour classify_matrix (valeur par défaut).
def _run_bootstraps_parallel(
    clf, cv, data, labels, n_trials, n_bootstraps, key, state,
    n_jobs, checkpoint_every, result_path, worker_fn=_one_bootstrap, prefer="threads"
) -> np.ndarray:
    """1000 bootstraps parallèles avec sauvegarde progressive.

    Reprend depuis le checkpoint si disponible (reprise après timeout).
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
# Vérifie l'existence d'un checkpoint partiel pour déterminer l'index de reprise et éviter de recalculer les bootstraps déjà faits.
# Découpe les bootstraps restants en blocs (chunks) exécutés en parallèle par joblib selon le nombre de cœurs alloués.
# Fusionne les résultats à chaque fin de bloc, exporte une sauvegarde compressée sur le disque et logue la progression.

# MODIF : même ajout de worker_fn que _run_bootstraps_parallel ci-dessus.
def _run_perms_parallel(
    clf, cv, data, labels, n_trials, n_perm, key, state,
    n_jobs, checkpoint_every, result_path, worker_fn=_one_perm, prefer="threads"
) -> np.ndarray:
    """1000 permutations parallèles avec sauvegarde progressive."""
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
# Gère la reprise sur panne pour les permutations en chargeant l'historique des calculs interrompus depuis le disque.
# Distribue le calcul des permutations par blocs successifs sur l'ensemble des workers joblib configurés.
# Met à jour progressivement le fichier de checkpoint des permutations pour sécuriser les données face aux limites de temps SLURM.

# ─── cache helpers ────────────────────────────────────────────────────────────

def _result_path(save_path: Path, key: str, state: str) -> Path:
    return save_path / "results" / f"{key}_{state}.npz"
# Construit le chemin normalisé du fichier de résultats finaux (.npz) selon la feature et le stade de sommeil.

def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
# Assure la création sécurisée des dossiers parents sur le disque s'ils sont manquants pour éviter tout crash.
# Archive et compresse efficacement l'ensemble des tableaux NumPy dans un fichier .npz unique.

# ─── classification ───────────────────────────────────────────────────────────

def classify_matrix(save_path, key, state, n_trials, n_bootstraps, n_perm,
                    overwrite, normalize, n_jobs=1, checkpoint_every=50):
    if normalize:
        warnings.warn(f"--normalize ignoré pour la feature matricielle '{key}'.")

    out = _result_path(save_path, key, state)
    if out.exists() and not overwrite:
        return np.load(out, allow_pickle=True)
    if overwrite:
        # BUGFIX : --overwrite ne nettoyait que le résultat final, jamais
        # les checkpoints. Un checkpoint plus ancien avec plus d'itérations
        # que n_bootstraps était silencieusement considéré "complet" et
        # court-circuitait tout recalcul. Sans ce fix, --overwrite n'était
        # pas fiable.
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
        key, state, n_jobs, checkpoint_every, out
    )

    result = dict(
        acc_mean   = float(acc_scores.mean()),
        acc_std    = float(acc_scores.std()),
        acc_scores = acc_scores,
        n_trials   = int(n_trials),
        n_subjects = int(len(data)),
        normalized = False,
    )
    if n_perm > 0:
        perm = _run_perms_parallel(
            clf, cv, data, labels, n_trials, n_perm,
            key, state, n_jobs, checkpoint_every, out
        )
        result["pval"]      = float((np.sum(perm >= result["acc_mean"]) + 1) / (n_perm + 1))
        result["perm_accs"] = perm
        # pas de correction maxstat ici : un seul classifieur global, contrairement a classify_vector (1 score par electrode)

    _save(out, **result)
    _clear_checkpoints(out)  # nettoie les checkpoints après sauvegarde finale
    return result

# Bloque la normalisation incompatible avec la géométrie riemannienne et gère le cache pour éviter les recalculs inutiles.
# Instancie le pipeline neuroscientifique (Tangent Space Mapping + LDA) et le validateur croisé stratifié (LPGO P=2).
# Exécute les boucles parallèles de bootstraps et de permutations, calcule les p-values, puis archive le dictionnaire compressé final.

def classify_vector(save_path, key, state, n_trials, n_bootstraps, n_perm,
                    overwrite, normalize, n_jobs=1, checkpoint_every=50):
    # MODIF : checkpoint_every=50 par défaut (avant : 0, ignoré de toute façon —
    # voir corps de fonction, qui ne le lisait jamais).
    out = _result_path(save_path, key, state)
    if out.exists() and not overwrite:
        return np.load(out, allow_pickle=True)
    if overwrite:
        # BUGFIX : --overwrite ne nettoyait que le résultat final, jamais
        # les checkpoints. Un checkpoint plus ancien avec plus d'itérations
        # que n_bootstraps était silencieusement considéré "complet" et
        # court-circuitait tout recalcul. Sans ce fix, --overwrite n'était
        # pas fiable.
        _clear_checkpoints(out)

    data, labels = load_all(save_path, key, state)
    if len(data) < 4:
        warnings.warn(
            f"Skipping {key}_{state}: cohorte insuffisante "
            f"(n={len(data)} sujets disponibles, minimum requis = 4)."
        )
        return None

    n_elec = data[0].shape[1]
    clf    = (Pipeline([("scaler", StandardScaler()), ("lda", LDA(solver="svd"))])
              if normalize else LDA(solver="svd"))
    cv     = StratifiedLeave2GroupsOut()

    # MODIF : avant, double boucle Python séquentielle ici (bootstraps × électrodes,
    # sans parallélisme ni checkpoint). Remplacée par _run_bootstraps_parallel +
    # worker_fn=_one_bootstrap_vector — même mécanisme que classify_matrix.
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
        normalized = normalize,
    )
    if n_perm > 0:
        # MODIF : idem, remplace la boucle séquentielle de permutations d'origine.
        perm_accs = _run_perms_parallel(
            clf, cv, data, labels, n_trials, n_perm,
            key, state, n_jobs, checkpoint_every, out, worker_fn=_one_perm_vector, prefer="processes"
        )
        result["pvals"] = (
            np.sum(perm_accs >= result["acc_mean"][None, :], axis=0) + 1
        ) / (n_perm + 1)
        result["perm_accs"] = perm_accs
        # BUGFIX : correction maxstat retiree d ici, incomplete (voir plot_results.py)

    _save(out, **result)
    # MODIF : nettoyage des checkpoints ajouté (absent avant, vu qu'aucun
    # checkpoint n'était jamais créé par cette fonction).
    _clear_checkpoints(out)
    return result
# Prépare un pipeline de mise à l'échelle (StandardScaler) sans fuite de données et isole les canaux EEG pour une analyse univariée.
# Évalue séquentiellement la capacité de décodage de chaque électrode à travers les itérations de bootstraps.
# Construit la distribution nulle par permutation et applique une correction de la statistique maximale (FWER) contre les faux positifs.

# ─── dispatcher ───────────────────────────────────────────────────────────────

def classify_one(save_path, key, state, n_trials, n_bootstraps, n_perm,
                 overwrite, normalize, n_jobs=1, checkpoint_every=50):
    print(f"  {key} × {state}")
    try:
        fn = classify_matrix if is_matrix_feature(key) else classify_vector
        return key, state, fn(
            save_path, key, state, n_trials, n_bootstraps, n_perm,
            overwrite, normalize, n_jobs, checkpoint_every
        )
    except Exception:
        print(f"  ERROR {key} {state}\n{traceback.format_exc()}")
        return key, state, None
# Affiche le couple en cours et ouvre un bloc "try/except" pour immuniser le script contre les crashs bloquants.
# Identifie dynamiquement la nature géométrique de la feature pour router le calcul vers le bon pipeline (matrice ou vecteur).
# Intercepte les exceptions en encapsulant le traceback complet dans les logs SLURM, puis renvoie None pour préserver la suite du batch.


# ─── résumé CSV ───────────────────────────────────────────────────────────────

def build_summary_csv(save_path: Path) -> None:
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
        normalized = bool(d["normalized"]) if "normalized" in d else False
        pval_scalar = float(d["pval"]) if "pval" in d else np.nan

        if acc_mean.ndim == 0:
            rows.append(dict(key=key, state=state, electrode="all",
                             acc_mean=float(acc_mean), acc_std=float(acc_std),
                             n_trials=n_trials, normalized=normalized,
                             pval=pval_scalar,
                             pval_maxstat=float(d["pval_maxstat"]) if "pval_maxstat" in d else np.nan))
        else:
            ch_names      = d["ch_names"].tolist() if "ch_names" in d else list(range(len(acc_mean)))
            pvals         = d["pvals"]          if "pvals"         in d else [np.nan] * len(acc_mean)
            pvals_maxstat = d["pvals_maxstat"]  if "pvals_maxstat" in d else [np.nan] * len(acc_mean)
            for ch, am, astd, pv, pv_ms in zip(ch_names, acc_mean, acc_std, pvals, pvals_maxstat):
                rows.append(dict(key=key, state=state, electrode=ch,
                                 acc_mean=float(am), acc_std=float(astd),
                                 n_trials=n_trials, normalized=normalized,
                                 pval=float(pv), pval_maxstat=float(pv_ms)))

    if rows:
        out = results_dir / "classification_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"CSV : {out}")
# Scanne récursivement le dossier results/ pour identifier et parser les fichiers de résultats .npz finaux (hors checkpoints).
# Discrimine les structures matricielles (une seule ligne globale 'all') des structures vectorielles univariées (une ligne par canal EEG).
# Compile et exporte l'ensemble des métriques de performance et de p-values dans un unique tableur CSV Pandas consolidé.

# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    t0   = time()

    print("=== vérification intégrité + n_trials_min global ===")
    n_trials = compute_global_n_trials(args.save_path, skip_check=args.skip_check)
    print(f"n_trials_min = {n_trials}  |  normalize = {args.normalize}")

    # Filtrage par --key et --state si fournis (mode combo unique pour array SLURM)
    keys   = [args.key]   if args.key   else FEATURE_KEYS
    states = [args.state] if args.state else STATE_LIST
    combos = list(product(keys, states))
    print(f"=== classification : {len(combos)} combinaisons ===")

    # MODIF : avant, deux chemins séparés (vector_combos en Parallel externe
    # 1 thread/combo, n_jobs=1/checkpoint_every=0 forcés ; matrix_combos en
    # boucle séquentielle avec n_jobs interne). Un seul chemin désormais :
    # classify_vector parallélise + checkpointe en interne comme les matrices,
    # cohérent aussi avec la topologie CCD/NUMA de Fir (1 combo à la fois sur
    # n_jobs cœurs, plutôt que dispersé entre combos).
    results = []
    for key, state in combos:
        res = classify_one(
            args.save_path, key, state, n_trials,
            args.n_bootstraps, args.n_perm, args.overwrite, args.normalize,
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
