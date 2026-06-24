"""Analyse multi-feature : sélection de bandes par ROI et stade (thèse §1.3.3, Fig 5).

Reproduit le troisième résultat majeur du chapitre 1 d'Arthur, absent du pipeline
de base (classify.py ne fait que du single-feature). Pour chaque ROI et chaque
stade, on cherche quelle(s) bande(s) de fréquence l'algorithme sélectionne le plus
souvent pour décoder HR vs LR — d'où le résultat phare d'Arthur : delta+sigma en
préfrontal pendant S2.

Méthode (thèse §1.2.7 "Multi-Feature analysis") :
  - Features candidates par ROI = la PSD moyennée sur les électrodes du ROI, pour
    chacune des 5 bandes (delta, theta, alpha, sigma, beta) -> 5 features scalaires.
  - Exhaustive Feature Selection (mlxtend EFS) : évalue TOUTES les combinaisons de
    bandes (2^5 - 1 = 31) et retient la meilleure pour décoder LR vs HR.
  - Scoring dans une CV nichée : EFS interne, score sur un holdout stratifié de
    2 sujets (1 HR + 1 LR). Répété sur toutes les combinaisons de holdout
    (StratifiedLeave2GroupsOut), puis comptage du Selection Rate (SR) : fréquence à
    laquelle chaque combinaison de bandes est choisie comme meilleure.
  - Sortie : pour chaque (ROI, stade), la table des SR par combinaison de bandes +
    l'accuracy moyenne sur holdout de la combinaison la plus fréquente.

NB ROIs : le mapping ci-dessous suit le système 10-20 standard. Les ROIs EXACTS
d'Arthur ne sont pas documentés dans les fichiers fournis -> mapping proposé,
à confirmer avec Arthur (via Karim) avant publication. Le résultat délta+sigma
préfrontal d'Arthur porte sur Fp1/Fp2 (frontopolaire), cohérent avec PREFRONTAL ici.

Usage :
    python classify_multifeature.py \\
        --save-path /home/alouis/scratch/dream_features \\
        --n-jobs    $SLURM_CPUS_PER_TASK \\
        --n-bootstraps 200

Coût : EFS exhaustif (31 combos) × n_holdout (324) × n_bootstraps. Réduire
--n-bootstraps (ex 100-200) si trop lent ; l'EFS exhaustif sur 5 bandes est léger.
"""

import argparse
import traceback
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

from config_v3 import (
    CH_NAMES, N_EEG, FREQ_DICT,
    CLASSIFICATION_GROUPS, STATE_LIST,
    SUBJECT_LIST_ORDERED, SUBJECT_LABELS,
)
from utils import load_atomic

BANDS = list(FREQ_DICT)  # ['delta','theta','alpha','sigma','beta']

# ─── ROIs (système 10-20 standard ; à confirmer avec Arthur) ──────────────────
ROIS = {
    "prefrontal":     ["Fp1", "Fp2"],
    "fronto-central": ["Fz", "F3", "F4", "FC1", "FC2"],
    "temporal":       ["T3", "T4"],
    "centro-parietal":["Cz", "C3", "C4", "CP1", "CP2", "Pz", "P3", "P4"],
    "occipital":      ["O1", "O2"],
}
_CH_INDEX = {ch: i for i, ch in enumerate(CH_NAMES[:N_EEG])}
ROI_INDICES = {roi: [_CH_INDEX[c] for c in chs] for roi, chs in ROIS.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path",    type=Path, required=True)
    p.add_argument("--n-jobs",       type=int,  default=1)
    p.add_argument("--n-bootstraps", type=int,  default=200)
    p.add_argument("--overwrite",    action="store_true", default=False)
    return p.parse_args()


# ─── chargement : PSD par bande agrégée par ROI ───────────────────────────────

def load_psd_by_band(save_path, state):
    """Retourne {band: [array(n_epochs, 19) par sujet]} et labels alignés.

    Charge psd_{band} (vecteur par électrode) pour chaque sujet/stade.
    """
    per_band = {b: [] for b in BANDS}
    labels = []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        stages = CLASSIFICATION_GROUPS[state]
        ok = True
        sub_band = {}
        for b in BANDS:
            parts = [a for s in stages
                     if (a := load_atomic(save_path, f"psd_{b}", sub_id, s)) is not None]
            if not parts:
                ok = False
                break
            sub_band[b] = np.concatenate(parts, axis=0)  # (n_epochs, 19)
        if ok:
            for b in BANDS:
                per_band[b].append(sub_band[b])
            labels.append(label)
    return per_band, np.array(labels)


def roi_features(per_band, roi):
    """Pour un ROI : {band: [array(n_epochs,) par sujet]} (moyenne sur électrodes du ROI)."""
    idx = ROI_INDICES[roi]
    return {b: [arr[:, idx].mean(axis=1) for arr in per_band[b]] for b in BANDS}


# ─── bootstrap (réutilise la logique de classify.py) ──────────────────────────

def bootstrap_roi(roi_band, labels, n_trials, seed):
    """Construit X (n_samples, 5 bandes), y, groups par sous-échantillonnage équilibré."""
    rng = np.random.RandomState(seed)
    n_sub = len(labels)
    Xs, ys, gs = [], [], []
    for g in range(n_sub):
        n_ep = len(roi_band[BANDS[0]][g])
        if n_ep < n_trials:
            raise RuntimeError(f"groupe {g}: {n_ep} epochs < n_trials={n_trials}")
        sel = rng.choice(n_ep, size=n_trials, replace=False)
        feat = np.column_stack([roi_band[b][g][sel] for b in BANDS])  # (n_trials, 5)
        Xs.append(feat)
        ys.extend([labels[g]] * n_trials)
        gs.extend([g] * n_trials)
    return np.concatenate(Xs), np.array(ys), np.array(gs)


# ─── CV stratifiée LPGO P=2 (1 HR + 1 LR en test) ─────────────────────────────

def stratified_l2go_splits(y, groups):
    y, groups = np.asarray(y), np.asarray(groups)
    classes = np.unique(y)
    idx_per_cls = [np.where(y == c)[0] for c in classes]
    iters = [list(LeavePGroupsOut(1).split(np.arange(len(idx)), y[idx], groups[idx]))
             for idx in idx_per_cls]
    out = []
    for s0, s1 in product(iters[0], iters[1]):
        tr = np.concatenate([idx_per_cls[0][s0[0]], idx_per_cls[1][s1[0]]])
        te = np.concatenate([idx_per_cls[0][s0[1]], idx_per_cls[1][s1[1]]])
        out.append((tr, te))
    return out


# ─── EFS exhaustif sur les 5 bandes ───────────────────────────────────────────

def best_band_combo(X, y, splits):
    """Évalue toutes les combinaisons non vides de bandes, retourne (meilleure combo, acc).

    EFS « maison » (pas mlxtend) pour contrôler exactement la CV stratifiée par
    sujet. 31 combinaisons sur 5 bandes -> trivial. clf = LDA (cohérent thèse/Arthur).
    """
    n_bands = X.shape[1]
    clf = LDA(solver="svd")
    best_combo, best_acc = None, -1.0
    for r in range(1, n_bands + 1):
        for combo in _combinations(range(n_bands), r):
            cols = list(combo)
            accs = []
            for tr, te in splits:
                m = clone(clf).fit(X[np.ix_(tr, cols)], y[tr])
                accs.append(accuracy_score(y[te], m.predict(X[np.ix_(te, cols)])))
            acc = float(np.mean(accs))
            if acc > best_acc:
                best_acc, best_combo = acc, combo
    return best_combo, best_acc


def _combinations(iterable, r):
    from itertools import combinations
    return combinations(iterable, r)


def combo_name(combo):
    return "+".join(BANDS[i] for i in combo)


# ─── analyse d'un (ROI, stade) ────────────────────────────────────────────────

def analyze_roi_state(save_path, state, roi, per_band, labels, n_trials, n_bootstraps):
    roi_band = roi_features(per_band, roi)
    combo_counts = {}      # combo_name -> nb de fois choisie
    combo_accs   = {}      # combo_name -> liste des accuracies
    for i in range(n_bootstraps):
        seed = (hash((roi, state, i)) & 0xFFFFFFFF)
        X, y, groups = bootstrap_roi(roi_band, labels, n_trials, seed)
        splits = stratified_l2go_splits(y, groups)
        combo, acc = best_band_combo(X, y, splits)
        name = combo_name(combo)
        combo_counts[name] = combo_counts.get(name, 0) + 1
        combo_accs.setdefault(name, []).append(acc)

    rows = []
    for name, cnt in combo_counts.items():
        rows.append(dict(
            roi=roi, state=state, bands=name,
            selection_rate=cnt / n_bootstraps,
            acc_mean=float(np.mean(combo_accs[name])),
            acc_std=float(np.std(combo_accs[name])),
            n_selected=cnt,
        ))
    return rows


def _worker(save_path, state, roi, n_trials, n_bootstraps):
    print(f"  {roi} × {state}")
    try:
        per_band, labels = load_psd_by_band(save_path, state)
        if len(labels) < 4:
            return []
        return analyze_roi_state(save_path, state, roi, per_band, labels,
                                 n_trials, n_bootstraps)
    except Exception:
        print(f"  ERROR {roi} {state}\n{traceback.format_exc()}")
        return []


# ─── n_trials_min (réutilise la logique globale de classify.py) ───────────────

def compute_n_trials(save_path):
    counts = []
    for state in STATE_LIST:
        for sub_id in SUBJECT_LIST_ORDERED:
            stages = CLASSIFICATION_GROUPS[state]
            parts = [a for s in stages
                     if (a := load_atomic(save_path, "cov", sub_id, s)) is not None]
            if parts:
                counts.append(sum(len(p) for p in parts))
    if not counts:
        raise RuntimeError("Aucun .npz 'cov' — feat_extract complet ?")
    return int(min(counts))


def main():
    args = parse_args()
    t0 = time()

    out = args.save_path / "results" / "multifeature_summary.csv"
    if out.exists() and not args.overwrite:
        print(f"{out} existe déjà (--overwrite pour recalculer).")
        return

    n_trials = compute_n_trials(args.save_path)
    print(f"n_trials_min = {n_trials}")

    combos = list(product(STATE_LIST, ROIS))
    print(f"=== multi-feature EFS : {len(combos)} (ROI × stade) ===")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_worker)(args.save_path, state, roi, n_trials, args.n_bootstraps)
        for state, roi in combos
    )

    rows = [r for sub in results for r in sub]
    if not rows:
        print("Aucun résultat (feat_extract incomplet ?).")
        return

    df = pd.DataFrame(rows).sort_values(
        ["state", "roi", "selection_rate"], ascending=[True, True, False]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV : {out}")

    # résumé : meilleure combo (SR max) par ROI × stade, filtré SR >= 25% (seuil Arthur)
    print("\n=== combinaison dominante par ROI × stade (SR >= 25%) ===")
    for (state, roi), g in df.groupby(["state", "roi"]):
        top = g.iloc[0]
        if top["selection_rate"] >= 0.25:
            print(f"  {state:5s} {roi:16s} : {top['bands']:20s} "
                  f"SR={top['selection_rate']*100:5.1f}%  acc={top['acc_mean']*100:.2f}%")

    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal : {m}m{s:02d}s")


if __name__ == "__main__":
    main()