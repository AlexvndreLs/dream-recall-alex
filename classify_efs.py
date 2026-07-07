"""Analyse EFS ciblee sur les features survivantes (extension du chapitre 1, §1.3.3).

Contexte
--------
Le chapitre 1 d'Arthur (classify_multifeature.py dans ce repo) fait une Exhaustive
Feature Selection sur les 5 BANDES PSD, par ROI. Ce script etend l'idee : au lieu
de chercher parmi des bandes, il cherche parmi les FEATURES HETEROGENES qui ont
deja prouve porter du signal (survivantes), et regarde si les COMBINER ameliore le
decodage HR vs LR.

Les survivantes ne sont pas hard-codees : elles sont lues au runtime depuis
results/pvalue_summary_table.csv (produit par build_pvalue_summary_table.py) en
filtrant p_non_corrige_subject < --alpha. Si le CSV change, la liste suit.

Deux passes
-----------
1. INTRA-ETAT (propre) : pour chaque etat (S2, SWS, ...), EFS sur les survivantes
   DE CET ETAT, sur les memes epochs. Les matriciels (cov, cosp_*) sont projetes
   en tangent space DANS le fold (TSClassifier), les vectoriels (psd_*, psd_osc_*,
   aperiodic, complexite) contribuent leur vecteur 19-electrodes. Chaque feature =
   un bloc atomique du vecteur final (pas d'eclatement des coords tangentes, cf
   litterature : les coords tangentes vivent ensemble, on ne les selectionne pas
   separement).

2. CROSS-ETAT (exploratoire) : agrege 1 vecteur par sujet (moyenne sur epochs) pour
   pouvoir combiner des features de STADES DIFFERENTS (ex cosp_sigma/S2 + psd_osc_delta/SWS)
   qui ne vivent pas sur les memes epochs. N tombe a 36 sujets -> resolution faible,
   marque exploratoire.

p-value (Option B)
------------------
Permutation AU NIVEAU SUJET (RFX, Combrisson & Jerbi 2015, correct) + RE-SELECTION
EFS COMPLETE sous chaque permutation : a chaque perm, on permute les labels sujets
et on refait toute la recherche EFS, de sorte que la p-value teste bien la procedure
de selection, pas seulement une combinaison figee.

/!\\ BIAIS DE SELECTION (double-dipping) /!\\
Les features testees ont ete choisies PARCE QU'ELLES survivent deja au test, sur
LES MEMES donnees. La p-value EFS est donc optimiste : ce n'est pas un test
d'hypothese independant mais une analyse EXPLORATOIRE post-hoc. Cet avertissement
est ecrit automatiquement dans le CSV de sortie et le log. Ne pas rapporter ces
p-values comme des decouvertes independantes.

Reutilise les primitives eprouvees de classify.py (bootstrap_sample,
StratifiedLeave2GroupsOut, permute_subject_labels, _seed, load_all) pour rester
coherent avec le reste du pipeline (memes seeds, meme CV, meme bootstrap).

Usage
-----
    python classify_efs.py \\
        --save-path   /scratch/alouis/dream_features_noica_1000hz_overlap \\
        --n-jobs      $SLURM_CPUS_PER_TASK \\
        --n-perm      199 \\
        --n-bootstraps 200 \\
        --alpha       0.05 \\
        --mode        intra          # ou 'cross', ou 'both'

Coût : domine par la projection tangent space des matriciels (~0.5s/fold/matriciel,
324 folds). Le cache par fold (project_fold_cache) projette chaque matriciel UNE
fois par fold et reutilise pour toutes les combinaisons -> gain ~x30 vs naif.
"""

import argparse
import traceback
import warnings
from itertools import combinations, product
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from pyriemann.tangentspace import TangentSpace

from config_v3 import (
    CLASSIFICATION_GROUPS,
    STATE_LIST,
    SUBJECT_LABELS,
    SUBJECT_LIST_ORDERED,
)
# Primitives reutilisees telles quelles depuis classify.py (source unique de verite :
# memes seeds, meme CV stratifiee, meme bootstrap que le pipeline single-feature).
from classify import (
    StratifiedLeave2GroupsOut,
    _seed,
    bootstrap_sample,
    is_matrix_feature,
    load_all,
    load_subject,
    permute_subject_labels,
)

PERM_SEED_OFFSET = 100_003  # identique a classify.py pour coherence des seeds


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path",     type=Path, required=True)
    p.add_argument("--n-jobs",        type=int,  default=1)
    p.add_argument("--n-perm",        type=int,  default=199)
    p.add_argument("--n-bootstraps",  type=int,  default=200)
    p.add_argument("--alpha",         type=float, default=0.05,
                   help="Seuil sur p_non_corrige_subject pour definir les survivantes.")
    p.add_argument("--max-features",  type=int,  default=3,
                   help="Taille max d'une combinaison EFS (defaut 3, cf interpretabilite).")
    p.add_argument("--mode",          type=str,  default="both",
                   choices=["intra", "cross", "both"])
    p.add_argument("--pval-col",      type=str,  default="p_non_corrige_subject",
                   help="Colonne du CSV a seuiller pour selectionner les survivantes.")
    p.add_argument("--survivors",     type=str,  default=None,
                   help="Override manuel : 'cosp_sigma/S2,psd_sigma/S2,...' "
                        "(court-circuite la lecture du CSV).")
    p.add_argument("--overwrite",     action="store_true", default=False)
    return p.parse_args()


# ─── selection des survivantes depuis le CSV ──────────────────────────────────

def load_survivors(save_path: Path, alpha: float, pval_col: str,
                   override: str | None) -> list[tuple[str, str]]:
    """Retourne [(feature, state), ...] avec pval_col < alpha.

    Lit results/pvalue_summary_table.csv. Ne hard-code rien : si le CSV evolue,
    la liste suit. --survivors permet un override manuel explicite.
    """
    if override:
        pairs = []
        for tok in override.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "/" not in tok:
                raise ValueError(f"--survivors : format attendu feature/state, recu '{tok}'")
            feat, state = tok.split("/", 1)
            pairs.append((feat.strip(), state.strip()))
        print(f"Survivantes (override manuel) : {len(pairs)}")
        return pairs

    csv_path = save_path / "results" / "pvalue_summary_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Lance build_pvalue_summary_table.py d'abord, "
            f"ou passe --survivors 'feat/state,...' en override."
        )
    df = pd.read_csv(csv_path)
    if pval_col not in df.columns:
        raise KeyError(f"Colonne '{pval_col}' absente de {csv_path}. "
                       f"Colonnes : {list(df.columns)}")
    df["_p"] = pd.to_numeric(df[pval_col], errors="coerce")  # PENDING/N/A -> NaN
    surv = df[df["_p"] < alpha][["feature", "state", "_p"]].copy()
    surv = surv.sort_values(["state", "_p"])
    pairs = list(zip(surv["feature"], surv["state"]))
    print(f"Survivantes ({pval_col} < {alpha}) : {len(pairs)}")
    for feat, state, pv in surv.itertuples(index=False):
        print(f"    {feat:16s} {state:5s}  {pval_col}={pv:.4f}")
    return pairs


# ─── projection tangent space cachee par fold ─────────────────────────────────

def project_fold_cache(feature_data: dict, splits: list) -> list:
    """Pre-projette chaque feature une fois par fold, reutilisable par tout combo.

    feature_data : {feature_key: array}
        - matriciel : (n_samples, 19, 19) SPD
        - vectoriel : (n_samples, 19)
    Retourne une liste (par fold) de (blocks_train, blocks_test) ou chaque bloc est
    deja projete (tangent pour les matriciels, brut pour les vectoriels).

    Anti-leakage : la TangentSpace est fittee sur le TRAIN du fold uniquement, jamais
    sur le test. Le cache ne fait que memoiser cette projection legitime pour eviter
    de la recalculer pour chaque combinaison.
    """
    cache = []
    for tr, te in splits:
        blocks_tr, blocks_te = {}, {}
        for key, arr in feature_data.items():
            if is_matrix_feature(key):
                ts = TangentSpace(metric="riemann")
                blocks_tr[key] = ts.fit_transform(arr[tr])  # fit sur train du fold
                blocks_te[key] = ts.transform(arr[te])
            else:
                blocks_tr[key] = arr[tr]
                blocks_te[key] = arr[te]
        cache.append((blocks_tr, blocks_te))
    return cache


def eval_combo_cached(combo: tuple, fold_cache: list, y: np.ndarray,
                      splits: list) -> float:
    """Accuracy moyenne CV d'une combinaison, en reutilisant le cache de projection."""
    accs = []
    for (blocks_tr, blocks_te), (tr, te) in zip(fold_cache, splits):
        Xtr = np.column_stack([blocks_tr[k] for k in combo])
        Xte = np.column_stack([blocks_te[k] for k in combo])
        m = LDA(solver="svd").fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], m.predict(Xte)))
    return float(np.mean(accs))


def run_efs(feature_data: dict, y: np.ndarray, groups: np.ndarray,
            feature_keys: list, max_features: int) -> tuple[tuple, float, dict]:
    """EFS exhaustif sur feature_keys (combinaisons de taille 1..max_features).

    Retourne (meilleure_combo, meilleure_acc, {combo: acc pour toutes}).
    """
    cv = StratifiedLeave2GroupsOut()
    splits = list(cv.split(feature_data[feature_keys[0]], y, groups))
    fold_cache = project_fold_cache(feature_data, splits)

    all_acc = {}
    best_combo, best_acc = None, -1.0
    rmax = min(max_features, len(feature_keys))
    for r in range(1, rmax + 1):
        for combo in combinations(feature_keys, r):
            acc = eval_combo_cached(combo, fold_cache, y, splits)
            all_acc[combo] = acc
            if acc > best_acc:
                best_acc, best_combo = acc, combo
    return best_combo, best_acc, all_acc


# ─── chargement des features d'un etat (intra) ────────────────────────────────

def load_state_features(save_path: Path, state: str,
                        feature_keys: list) -> tuple[dict, np.ndarray, np.ndarray]:
    """Charge toutes les features de `feature_keys` pour `state`, alignees sujet/epoch.

    Retourne (feature_data, labels_epoch, groups_epoch) APRES bootstrap-equilibrage
    -- non : ici on retourne les donnees BRUTES par sujet, le bootstrap est applique
    par l'appelant (une fois par bootstrap/perm). feature_data mappe key -> liste
    d'arrays par sujet ; labels = 1 label par sujet.
    """
    per_feat = {k: [] for k in feature_keys}
    labels = []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        arrs, ok = {}, True
        for k in feature_keys:
            a = load_subject(save_path, k, sub_id, state)
            if a is None:
                ok = False
                break
            arrs[k] = a
        # verifie l'alignement epoch entre features du meme sujet
        if ok:
            n0 = len(next(iter(arrs.values())))
            if any(len(a) != n0 for a in arrs.values()):
                ok = False
        if ok:
            for k in feature_keys:
                per_feat[k].append(arrs[k])
            labels.append(label)
    return per_feat, np.array(labels)


def bootstrap_multifeature(per_feat: dict, labels: np.ndarray, feature_keys: list,
                           n_trials: int, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Sous-echantillonnage equilibre PARTAGE entre toutes les features (memes epochs).

    Critique : le meme tirage d'indices est applique a toutes les features d'un
    sujet, sinon les blocs ne seraient plus alignes epoch-a-epoch dans le vecteur
    concatene. On tire donc les indices une fois par sujet, puis on indexe chaque
    feature avec.
    """
    rng = np.random.RandomState(seed)
    out = {k: [] for k in feature_keys}
    ys, gs = [], []
    for g in range(len(labels)):
        n_ep = len(per_feat[feature_keys[0]][g])
        if n_ep < n_trials:
            raise RuntimeError(f"groupe {g}: {n_ep} epochs < n_trials={n_trials}")
        idx = rng.choice(n_ep, size=n_trials, replace=False)
        for k in feature_keys:
            out[k].append(per_feat[k][g][idx])
        ys.extend([labels[g]] * n_trials)
        gs.extend([g] * n_trials)
    stacked = {k: np.concatenate(out[k], axis=0) for k in feature_keys}
    return stacked, np.array(ys), np.array(gs)


# ─── passe INTRA-ETAT ─────────────────────────────────────────────────────────

def analyze_intra_state(save_path, state, feature_keys, n_trials, n_bootstraps,
                        n_perm, max_features, n_jobs):
    """EFS intra-etat + Selection Rate + p-value option B (re-selection sous perm)."""
    print(f"\n=== INTRA {state} : features {feature_keys} ===")
    per_feat, labels = load_state_features(save_path, state, feature_keys)
    if len(labels) < 4:
        warnings.warn(f"{state}: cohorte insuffisante (n={len(labels)}), skip.")
        return None

    # ── bootstraps reels : Selection Rate + accuracy de la meilleure combo ──
    def _one_boot(i):
        fd, y, groups = bootstrap_multifeature(
            per_feat, labels, feature_keys, n_trials, _seed("efs", state, i)
        )
        combo, acc, _ = run_efs(fd, y, groups, feature_keys, max_features)
        return combo, acc

    boot_res = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_boot)(i) for i in range(n_bootstraps)
    )
    sel_counts, combo_accs = {}, {}
    for combo, acc in boot_res:
        name = "+".join(combo)
        sel_counts[name] = sel_counts.get(name, 0) + 1
        combo_accs.setdefault(name, []).append(acc)

    # combo dominante = Selection Rate max (la plus robuste aux tirages)
    dominant = max(sel_counts, key=sel_counts.get)
    dominant_acc = float(np.mean(combo_accs[dominant]))

    result = dict(
        state=state,
        feature_keys=list(feature_keys),
        n_subjects=int(len(labels)),
        n_trials=int(n_trials),
        selection_rates={k: v / n_bootstraps for k, v in sel_counts.items()},
        combo_acc_mean={k: float(np.mean(v)) for k, v in combo_accs.items()},
        dominant_combo=dominant,
        dominant_acc=dominant_acc,
    )

    # ── p-value option B : re-selection EFS complete sous permutation sujet ──
    if n_perm > 0:
        def _one_perm(p):
            labels_perm = permute_subject_labels(
                labels, _seed("perm", state, PERM_SEED_OFFSET + n_perm + p)
            )
            fd, y, groups = bootstrap_multifeature(
                per_feat, labels_perm, feature_keys, n_trials,
                _seed("perm", state, PERM_SEED_OFFSET + p)
            )
            # re-selection COMPLETE : on refait tout l'EFS, on garde l'acc de la
            # meilleure combo trouvee sous labels permutes (teste la procedure).
            _, best_acc_perm, _ = run_efs(fd, y, groups, feature_keys, max_features)
            return best_acc_perm

        perm_accs = np.array(Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_one_perm)(p) for p in range(n_perm)
        ))
        # p-value : l'accuracy de la combo dominante reelle depasse-t-elle le max
        # obtenu par selection sous labels permutes ?
        result["perm_best_accs"] = perm_accs
        result["pval_efs"] = float(
            (np.sum(perm_accs >= dominant_acc) + 1) / (n_perm + 1)
        )
        result["n_perm"] = int(n_perm)

    return result


# ─── passe CROSS-ETAT (exploratoire) ──────────────────────────────────────────

def load_subject_aggregated(save_path, feat_state_pairs):
    """1 vecteur par sujet par (feature,state), moyenne sur epochs.

    Permet de combiner des features de STADES DIFFERENTS (impossible epoch-a-epoch).
    Matriciel -> on ne peut pas moyenner en tangent hors-fold sans reference ; on
    moyenne les matrices SPD par sujet (moyenne euclidienne, approximation
    exploratoire assumee) puis on vectorise le triangle sup. Vectoriel -> moyenne
    des 19 valeurs par sujet.

    Retourne (X (n_sub, d), labels (n_sub,)) sujets alignes ; None si un sujet
    manque une des features requises.
    """
    from utils import upper_tri
    rows, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        blocks, ok = [], True
        for feat, state in feat_state_pairs:
            a = load_subject(save_path, feat, sub_id, state)
            if a is None:
                ok = False
                break
            if is_matrix_feature(feat):
                mean_mat = a.mean(axis=0, keepdims=True)          # (1,19,19)
                blocks.append(upper_tri(mean_mat).ravel())        # triangle sup
            else:
                blocks.append(a.mean(axis=0))                     # (19,)
        if ok:
            rows.append(np.concatenate(blocks))
            labels.append(label)
    if len(rows) < 4:
        return None, None
    return np.vstack(rows), np.array(labels)


def analyze_cross_state(save_path, survivors, n_perm, max_features, n_jobs):
    """EFS cross-etat exploratoire sur donnees agregees par sujet (N=36).

    Chaque 'feature candidate' est une paire (feature,state). L'EFS cherche la
    meilleure combinaison de paires. LOSO (leave-2-subjects-out) sur 36 sujets.
    Resolution faible -> EXPLORATOIRE.
    """
    print(f"\n=== CROSS-ETAT (exploratoire, N sujets agreges) ===")
    if len(survivors) < 2:
        warnings.warn("Moins de 2 survivantes : cross-etat sans objet, skip.")
        return None

    # candidats = paires (feature,state), nommes "feature@state"
    pair_names = [f"{f}@{s}" for f, s in survivors]

    # charge le bloc agrege de chaque paire, par sujet
    # (on charge une fois par paire, puis on assemble par combo)
    from utils import upper_tri
    per_pair, labels_ref = {}, None
    for (feat, state), name in zip(survivors, pair_names):
        X_one, labels = load_subject_aggregated(save_path, [(feat, state)])
        if X_one is None:
            warnings.warn(f"cross-etat : {name} indisponible, exclu.")
            continue
        per_pair[name] = X_one
        labels_ref = labels if labels_ref is None else labels_ref
        if len(labels) != len(labels_ref):
            warnings.warn(f"cross-etat : {name} desaligne sujets, exclu.")
            per_pair.pop(name, None)

    names = list(per_pair)
    if len(names) < 2:
        warnings.warn("cross-etat : <2 paires exploitables apres chargement, skip.")
        return None

    y = labels_ref
    groups = np.arange(len(y))
    cv = StratifiedLeave2GroupsOut()
    splits = list(cv.split(per_pair[names[0]], y, groups))

    def eval_pair_combo(combo):
        accs = []
        for tr, te in splits:
            Xtr = np.column_stack([per_pair[n][tr] for n in combo])
            Xte = np.column_stack([per_pair[n][te] for n in combo])
            m = LDA(solver="svd").fit(Xtr, y[tr])
            accs.append(accuracy_score(y[te], m.predict(Xte)))
        return float(np.mean(accs))

    all_acc, best_combo, best_acc = {}, None, -1.0
    rmax = min(max_features, len(names))
    for r in range(1, rmax + 1):
        for combo in combinations(names, r):
            acc = eval_pair_combo(combo)
            all_acc[combo] = acc
            if acc > best_acc:
                best_acc, best_combo = acc, combo

    result = dict(
        candidates=names,
        n_subjects=int(len(y)),
        combo_acc={"+".join(c): a for c, a in all_acc.items()},
        best_combo="+".join(best_combo),
        best_acc=float(best_acc),
        exploratory=True,
    )

    # p-value option B sur donnees agregees (permutation sujet + re-selection)
    if n_perm > 0:
        def _one_perm(p):
            yp = permute_subject_labels(y, _seed("permx", "cross", PERM_SEED_OFFSET + p))
            bacc = -1.0
            for r in range(1, rmax + 1):
                for combo in combinations(names, r):
                    accs = []
                    for tr, te in splits:
                        Xtr = np.column_stack([per_pair[n][tr] for n in combo])
                        Xte = np.column_stack([per_pair[n][te] for n in combo])
                        m = LDA(solver="svd").fit(Xtr, yp[tr])
                        accs.append(accuracy_score(yp[te], m.predict(Xte)))
                    bacc = max(bacc, float(np.mean(accs)))
            return bacc

        perm_accs = np.array(Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_one_perm)(p) for p in range(n_perm)
        ))
        result["perm_best_accs"] = perm_accs
        result["pval_efs"] = float((np.sum(perm_accs >= best_acc) + 1) / (n_perm + 1))
        result["n_perm"] = int(n_perm)

    return result


# ─── n_trials_min (coherent avec classify.py, min sur cov) ────────────────────

def compute_n_trials(save_path: Path) -> int:
    counts = []
    for state in STATE_LIST:
        for sub_id in SUBJECT_LIST_ORDERED:
            a = load_subject(save_path, "cov", sub_id, state)
            if a is not None:
                counts.append(len(a))
    if not counts:
        raise RuntimeError("Aucun .npz 'cov' — feat_extract complet ?")
    return int(min(counts))


# ─── avertissement double-dipping (ecrit dans CSV + log) ──────────────────────

DOUBLE_DIP_WARNING = (
    "ANALYSE EXPLORATOIRE — BIAIS DE SELECTION (double-dipping) : les features "
    "testees ici ont ete pre-selectionnees parce qu'elles survivent deja au test "
    "de significativite, sur LES MEMES donnees. Les p-values EFS ci-dessous sont "
    "donc optimistes et ne constituent PAS un test d'hypothese independant. Ne pas "
    "les rapporter comme des decouvertes independantes. Pour un test non biaise, il "
    "faudrait une selection interne au fold sur les 16 features brutes (voir "
    "discussion : cout whole-head)."
)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time()

    print("!" * 80)
    print(DOUBLE_DIP_WARNING)
    print("!" * 80)

    results_dir = args.save_path / "results"
    out_csv = results_dir / "efs_survivors_summary.csv"
    if out_csv.exists() and not args.overwrite:
        print(f"{out_csv} existe deja (--overwrite pour recalculer).")
        return

    survivors = load_survivors(args.save_path, args.alpha, args.pval_col, args.survivors)
    if not survivors:
        print("Aucune survivante — rien a faire.")
        return

    n_trials = compute_n_trials(args.save_path)
    print(f"\nn_trials_min = {n_trials}")

    # regroupe les survivantes par etat pour la passe intra
    by_state: dict[str, list] = {}
    for feat, state in survivors:
        by_state.setdefault(state, []).append(feat)

    rows = []

    # ── passe INTRA ──
    if args.mode in ("intra", "both"):
        for state, feats in by_state.items():
            if len(feats) < 2:
                print(f"\n[INTRA {state}] 1 seule survivante ({feats}) -> "
                      f"pas de combinaison possible, skip intra (gardee pour cross).")
                continue
            res = analyze_intra_state(
                args.save_path, state, feats, n_trials,
                args.n_bootstraps, args.n_perm, args.max_features, args.n_jobs
            )
            if res is None:
                continue
            print(f"  [{state}] combo dominante : {res['dominant_combo']} "
                  f"(SR={res['selection_rates'][res['dominant_combo']]*100:.0f}%, "
                  f"acc={res['dominant_acc']*100:.2f}%"
                  + (f", p_efs={res['pval_efs']:.4f})" if 'pval_efs' in res else ")"))
            for combo_name, sr in sorted(res["selection_rates"].items(),
                                         key=lambda x: -x[1]):
                rows.append(dict(
                    mode="intra", state=state, combo=combo_name,
                    selection_rate=sr,
                    acc_mean=res["combo_acc_mean"][combo_name],
                    is_dominant=(combo_name == res["dominant_combo"]),
                    pval_efs=res.get("pval_efs") if combo_name == res["dominant_combo"] else "",
                    n_subjects=res["n_subjects"], n_trials=res["n_trials"],
                    exploratory=True,
                ))

    # ── passe CROSS ──
    if args.mode in ("cross", "both"):
        res = analyze_cross_state(
            args.save_path, survivors, args.n_perm, args.max_features, args.n_jobs
        )
        if res is not None:
            print(f"  [CROSS] meilleure combo : {res['best_combo']} "
                  f"(acc={res['best_acc']*100:.2f}%"
                  + (f", p_efs={res['pval_efs']:.4f})" if 'pval_efs' in res else ")"))
            for combo_name, acc in sorted(res["combo_acc"].items(), key=lambda x: -x[1]):
                rows.append(dict(
                    mode="cross", state="multi", combo=combo_name,
                    selection_rate="",
                    acc_mean=acc,
                    is_dominant=(combo_name == res["best_combo"]),
                    pval_efs=res.get("pval_efs") if combo_name == res["best_combo"] else "",
                    n_subjects=res["n_subjects"], n_trials="aggregated",
                    exploratory=True,
                ))

    if not rows:
        print("\nAucun resultat produit.")
        return

    df = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    # l'avertissement est ecrit en tete du CSV (ligne commentee) + colonne exploratory
    with open(out_csv, "w") as f:
        f.write(f"# {DOUBLE_DIP_WARNING}\n")
    df.to_csv(out_csv, mode="a", index=False)
    print(f"\nCSV : {out_csv}")
    print(df.to_string(index=False))

    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal : {m}m{s:02d}s")


if __name__ == "__main__":
    main()
