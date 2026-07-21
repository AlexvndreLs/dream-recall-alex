"""TEST AUTONOME — spec_entropy_osc : extraction + classification en un seul fichier.

But
---
Tester une feature candidate : l'entropie spectrale de Shannon calculee sur le
RESIDU OSCILLATOIRE FOOOF (flat_ratio = psd observee / fit aperiodic), et non
sur le spectre brut. Objectif scientifique (cf Aamodt et al. 2022) : mesurer si
la complexite spectrale apporte quelque chose AU-DELA de la pente 1/f, une fois
celle-ci retiree.

Pourquoi spec_entropy et PAS perm_entropy / higuchi_fd sur le residu
-------------------------------------------------------------------
perm_entropy et higuchi_fd lisent l'ORDRE TEMPOREL des echantillons (l'axe temps).
FOOOF opere en aval de Welch, qui a deja jete la phase en passant au spectre de
puissance. Le residu oscillatoire est donc un SPECTRE, sans axe temps recuperable :
reconstruire un temporel exigerait une phase que FOOOF n'a pas conservee (phase
aleatoire -> on mesure le RNG ; phase originale -> le 1/f revient par la bande).
spec_entropy, elle, ne lit jamais le temps : elle mesure la forme du spectre.
C'est la seule des trois dont l'objet mathematique survit au passage FOOOF.

Ce fichier est un TEST hors pipeline principal : il NE touche PAS a config_v3.py
ni a FEATURE_KEYS. Il ecrit ses .npz atomiques dans un sous-dossier dedie
(cle "spec_entropy_osc") sous --save-path, puis classe.

Methodo de classification : reproduction fidele de classify.py (mode vecteur) :
  - LDA(solver="svd") par electrode
  - StratifiedLeave2GroupsOut (LPGO P=2 stratifie HR/LR)
  - bootstrap : sous-tirage SANS remise a n_trials_min par sujet
  - permutation SUJET (RFX, Combrisson & Jerbi 2015)
  - correction max-stat sur les 19 electrodes (pooled null du max sur electrodes)

Usage :
    python test_spec_entropy_osc.py \\
        --deriv-path /scratch/alouis/dream_bids/derivatives/preprocessed-noica \\
        --save-path  /scratch/alouis/dream_features_noica_1000hz_overlap \\
        --n-jobs     $SLURM_CPUS_PER_TASK \\
        --n-perm     1000 \\
        --n-bootstraps 1000 \\
        [--overwrite-feat]   # force la re-extraction des .npz spec_entropy_osc
        [--state S2]         # restreint a un etat (defaut : les 4)

Sortie : results_spec_entropy_osc/spec_entropy_osc_<state>.npz + un CSV recap.
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
import mne
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeavePGroupsOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from specparam import SpectralGroupModel

from config_v3 import (
    SFREQ_PREPROC, PER_BLACKLIST_STR, JBE_SUBJECTS_STR,
    N_SAMPLES, N_EEG, CH_NAMES,
    WINDOW, OVERLAP, FOOOF_FREQ_RANGE,
    ATOMIC_STAGES, STAGE_LABEL_TO_ATOMIC,
    CLASSIFICATION_GROUPS, STATE_LIST,
    SUBJECT_LABELS, SUBJECT_LIST_ORDERED, SUBJECT_IDS,
)
from utils import load_atomic

SF        = int(SFREQ_PREPROC)
FEAT_KEY  = "spec_entropy_osc"
PERM_SEED_OFFSET = 100_003


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — EXTRACTION spec_entropy_osc
# ══════════════════════════════════════════════════════════════════════════════
# Reprend a l'identique la logique de feat_extract_umap_fooof_v4.py
# (segmentation atomique 30s, Welch, fit_fooof -> flat_ratio) mais ne calcule
# QUE spec_entropy_osc. Cache par sujet/stade atomique sous save_path/FEAT_KEY/.

def _vhdr(deriv_path: Path, sub_id: str) -> Path:
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_eeg.vhdr")


def _events(deriv_path: Path, sub_id: str) -> Path:
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_events.tsv")


def _choose_scorer(sub_id: str) -> str:
    if sub_id not in PER_BLACKLIST_STR:
        return "per"
    if sub_id in JBE_SUBJECTS_STR:
        return "jbe"
    raise ValueError(f"sub-{sub_id}: no valid scorer")


def load_epochs_by_atomic_stage(deriv_path: Path, sub_id: str) -> dict[str, np.ndarray]:
    """Copie exacte de feat_extract_umap_fooof_v4 : epochs 30s non-chevauchants
    par stade atomique. Returns dict[atomic] -> (n_epochs, 19, N_SAMPLES)."""
    raw = mne.io.read_raw_brainvision(_vhdr(deriv_path, sub_id), preload=True, verbose=False)
    raw.pick(CH_NAMES[:N_EEG])
    assert raw.info["sfreq"] == SFREQ_PREPROC, (
        f"sub-{sub_id}: sfreq fichier ({raw.info['sfreq']}) != SFREQ_PREPROC "
        f"({SFREQ_PREPROC}) — DECIMATE/SFREQ_PREPROC desynchronises dans config_v3.py."
    )
    n_total = raw.n_times

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"

    df = pd.read_csv(_events(deriv_path, sub_id), sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = (df[df["stage"].isin(STAGE_LABEL_TO_ATOMIC)]
          .sort_values("sample").reset_index(drop=True))

    epochs: dict[str, list[np.ndarray]] = {s: [] for s in ATOMIC_STAGES}
    i = 0
    while i + 29 < len(df):
        block   = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages  = block["stage"].values
        if not (np.all(samples == samples[0] + np.arange(30) * SF) and
                np.all(stages == stages[0])):
            i += 1
            continue
        end = int(samples[0]) + N_SAMPLES
        if end > n_total:
            raise ValueError(f"sub-{sub_id}: epoch depasse fin fichier (end={end}, n_total={n_total})")
        epoch = raw.get_data(start=int(samples[0]), stop=end)
        epochs[STAGE_LABEL_TO_ATOMIC[stages[0]]].append(epoch)
        i += 30
    return {s: np.stack(e) for s, e in epochs.items() if e}


def compute_psd_spectrum(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_ep, 19, N_SAMPLES) -> psds (n_ep, 19, n_freqs), freqs. Identique feat_extract."""
    return mne.time_frequency.psd_array_welch(
        data, sfreq=SF, fmin=FOOOF_FREQ_RANGE[0], fmax=FOOOF_FREQ_RANGE[1],
        n_fft=WINDOW, n_overlap=OVERLAP, n_per_seg=WINDOW, window="hann", verbose=False,
    )


def fit_fooof_flat_ratio(psds: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Fit FOOOF fixe -> flat_ratio (n_ep, 19, n_freqs) = psd / fit_aperiodic.

    Copie EXACTE du calcul flat_ratio de feat_extract_umap_fooof_v4.fit_fooof :
    ratio lineaire (psd observee / 10**ap_fit_log), facteur d'exces au-dessus
    du 1/f. On ne renvoie QUE flat_ratio (pas l'exposant, inutile ici).
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs)

    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)

    aperiodic = fg.get_params("aperiodic")
    offsets   = aperiodic[:, 0:1]
    exponents = aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]
    flat_ratio = flat_psds / (10 ** ap_fit_log)
    return flat_ratio.reshape(n_epochs, n_ch, n_freqs)


def spectral_entropy_normalized(spectrum: np.ndarray) -> np.ndarray:
    """(n_ep, 19, n_freqs) -> (n_ep, 19). Shannon normalise de la distribution
    de puissance le long de l'axe frequences.

    Meme formule EXACTE que compute_complexity dans feat_extract_umap_fooof_v4
    (branche spec_entropy), mais appliquee au residu oscillatoire flat_ratio.
    flat_ratio est positif (ratio de puissances) -> normalisation valide.
    """
    psd_sum  = spectrum.sum(axis=-1, keepdims=True)
    psd_norm = np.divide(spectrum, psd_sum, out=np.zeros_like(spectrum), where=psd_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_psd = np.where(psd_norm > 0, np.log2(psd_norm), 0.0)
    spec_ent = -(psd_norm * log_psd).sum(axis=-1)
    spec_ent /= np.log2(spectrum.shape[-1])
    return spec_ent


def extract_subject(deriv_path: Path, save_path: Path, sub_id: str,
                    overwrite_feat: bool) -> None:
    """Extrait spec_entropy_osc pour un sujet, cache par stade atomique."""
    if not _vhdr(deriv_path, sub_id).exists():
        print(f"sub-{sub_id}: derivative absent, skip")
        return

    # skip si tous les stades atomiques deja caches
    out_dir = save_path / FEAT_KEY
    if not overwrite_feat and all(
        (out_dir / f"{FEAT_KEY}_s{sub_id}_{st}.npz").exists()
        for st in ATOMIC_STAGES
    ):
        # certains sujets n'ont pas tous les stades ; on ne peut pas savoir
        # a priori lesquels existent sans relire. Verif souple : si au moins
        # un .npz existe et overwrite off, on considere le sujet fait.
        existing = list(out_dir.glob(f"{FEAT_KEY}_s{sub_id}_*.npz"))
        if existing:
            print(f"sub-{sub_id}: deja cache ({len(existing)} stades), skip")
            return

    try:
        atomic = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        print(f"sub-{sub_id}: ERROR loading\n{traceback.format_exc()}")
        return

    for stage, data in atomic.items():
        out = out_dir / f"{FEAT_KEY}_s{sub_id}_{stage}.npz"
        if out.exists() and not overwrite_feat:
            continue
        try:
            psds, freqs = compute_psd_spectrum(data)
            flat_ratio  = fit_fooof_flat_ratio(psds, freqs)
            se_osc      = spectral_entropy_normalized(flat_ratio)   # (n_ep, 19)
        except Exception:
            print(f"sub-{sub_id} {stage}: ERROR extract\n{traceback.format_exc()}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, data=se_osc)
        print(f"  sub-{sub_id} {stage}: {se_osc.shape[0]} epochs -> {out.name}")
    print(f"sub-{sub_id}: extraction done")


# ══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — CLASSIFICATION (reproduction fidele de classify.py, mode vecteur)
# ══════════════════════════════════════════════════════════════════════════════

def _seed(key: str, state: str, idx: int) -> int:
    h = md5(f"{key}_{state}_{idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def load_subject(save_path: Path, key: str, sub_id: str, state: str) -> np.ndarray | None:
    stages = CLASSIFICATION_GROUPS[state]
    parts  = [a for s in stages if (a := load_atomic(save_path, key, sub_id, s)) is not None]
    return np.concatenate(parts, axis=0) if parts else None


def load_all(save_path: Path, key: str, state: str) -> tuple[list, np.ndarray]:
    data, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        arr = load_subject(save_path, key, f"{sub_id:02d}", state)
        if arr is not None:
            data.append(arr)
            labels.append(label)
    return data, np.array(labels)


def compute_n_trials(save_path: Path, key: str, states: list[str]) -> int:
    """min du nombre d'epochs par sujet/stade DISPONIBLE pour cette feature.

    NB : dans classify.py, n_trials est calcule sur 'cov' (REF_KEY) pour garantir
    la comparabilite entre features. Ici, test isole : on prend le min sur la
    feature elle-meme, sur les etats testes. Si tu veux la MEME valeur n_trials
    que le pipeline principal, passe --force-n-trials.
    """
    counts = []
    for state in states:
        for sub_id in SUBJECT_LIST_ORDERED:
            arr = load_subject(save_path, key, f"{sub_id:02d}", state)
            if arr is not None:
                counts.append(len(arr))
    if not counts:
        raise RuntimeError(f"Aucun .npz {key} trouve — extraction faite ?")
    return int(min(counts))


def bootstrap_sample(data: list, labels: np.ndarray, n_trials: int, seed: int):
    rng = np.random.RandomState(seed)
    Xs, ys, gs = [], [], []
    for g, (arr, lab) in enumerate(zip(data, labels)):
        if len(arr) < n_trials:
            raise RuntimeError(f"Groupe {g}: {len(arr)} epochs < n_trials={n_trials}.")
        idx = rng.choice(len(arr), size=n_trials, replace=False)
        Xs.append(arr[idx]); ys.extend([lab] * n_trials); gs.extend([g] * n_trials)
    return np.concatenate(Xs), np.array(ys), np.array(gs)


def permute_subject_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """RFX : permute les labels HR/LR au niveau SUJET (Combrisson & Jerbi 2015)."""
    return np.random.RandomState(seed).permutation(labels)


class StratifiedLeave2GroupsOut:
    """LPGO P=2 stratifie : 1 sujet HR + 1 sujet LR en test. Copie classify.py."""
    def split(self, X, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Attendu 2 classes, obtenu {len(classes)}")
        idx_per_cls = [np.where(y == c)[0] for c in classes]
        iters = [list(LeavePGroupsOut(1).split(np.arange(len(idx)), y[idx], groups[idx]))
                 for idx in idx_per_cls]
        for s0, s1 in product(iters[0], iters[1]):
            yield (np.concatenate([idx_per_cls[0][s0[0]], idx_per_cls[1][s1[0]]]),
                   np.concatenate([idx_per_cls[0][s0[1]], idx_per_cls[1][s1[1]]]))

    def get_n_splits(self, X, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        n = 1
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            n *= LeavePGroupsOut(1).get_n_splits(None, y[idx], groups[idx])
        return n


def run_cv(clf, splits, X, y) -> float:
    return float(np.mean([
        accuracy_score(y[te], clone(clf).fit(X[tr], y[tr]).predict(X[te]))
        for tr, te in splits
    ]))


def _one_bootstrap_vector(clf, cv, data, labels, n_trials, key, state, i) -> np.ndarray:
    X, y, groups = bootstrap_sample(data, labels, n_trials, _seed(key, state, i))
    splits = list(cv.split(X, y, groups))
    n_elec = X.shape[1]
    return np.array([run_cv(clf, splits, X[:, e:e + 1], y) for e in range(n_elec)])


def _one_perm_vector(clf, cv, data, labels, n_trials, key, state, p, n_perm) -> np.ndarray:
    labels_perm = permute_subject_labels(labels, _seed('perm', state, PERM_SEED_OFFSET + n_perm + p))
    X, y, groups = bootstrap_sample(data, labels_perm, n_trials, _seed('perm', state, PERM_SEED_OFFSET + p))
    splits = list(cv.split(X, y, groups))
    n_elec = X.shape[1]
    return np.array([run_cv(clf, splits, X[:, e:e + 1], y) for e in range(n_elec)])


def classify_vector(save_path, key, state, n_trials, n_bootstraps, n_perm,
                    normalize, n_jobs, prefer="processes"):
    data, labels = load_all(save_path, key, state)
    if len(data) < 4:
        warnings.warn(f"Skip {key}_{state}: cohorte insuffisante (n={len(data)}).")
        return None

    clf = (Pipeline([("scaler", StandardScaler()), ("lda", LDA(solver="svd"))])
           if normalize else LDA(solver="svd"))
    cv  = StratifiedLeave2GroupsOut()

    acc_scores = np.array(Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_one_bootstrap_vector)(clf, cv, data, labels, n_trials, key, state, i)
        for i in range(n_bootstraps)
    ))  # (n_bootstraps, n_elec)

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
        perm_accs = np.array(Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_one_perm_vector)(clf, cv, data, labels, n_trials, key, state, p, n_perm)
            for p in range(n_perm)
        ))  # (n_perm, n_elec)

        # p-values non corrigees (par electrode)
        result["pvals"] = (np.sum(perm_accs >= result["acc_mean"][None, :], axis=0) + 1) / (n_perm + 1)
        result["perm_accs"] = perm_accs

        # correction max-stat (FWER) : distribution nulle du MAX sur electrodes.
        # Reproduit la logique de compute_maxstat_correction.py (pooled null),
        # ici en local sur les 19 electrodes de cette seule feature.
        null_max = perm_accs.max(axis=1)  # (n_perm,)
        result["pvals_maxstat"] = np.array([
            (np.sum(null_max >= am) + 1) / (n_perm + 1) for am in result["acc_mean"]
        ])
        result["null_max"] = null_max

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine derivative preprocessed (ex: .../derivatives/preprocessed-noica)")
    p.add_argument("--save-path",  type=Path, required=True,
                   help="Dossier features (les .npz spec_entropy_osc y seront caches)")
    p.add_argument("--n-jobs",     type=int, default=1)
    p.add_argument("--n-perm",     type=int, default=1000)
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--state",      type=str, default=None,
                   help="Etat unique (S2/SWS/REM/NREM). Defaut : les 4.")
    p.add_argument("--normalize",  action="store_true", default=False)
    p.add_argument("--force-n-trials", type=int, default=None,
                   help="Force n_trials (sinon min sur la feature/etats testes).")
    p.add_argument("--overwrite-feat", action="store_true", default=False,
                   help="Force la re-extraction des .npz spec_entropy_osc.")
    p.add_argument("--skip-extract", action="store_true", default=False,
                   help="Saute l'extraction (suppose les .npz deja presents).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time()
    states = [args.state] if args.state else STATE_LIST

    # ── PARTIE 1 : extraction ────────────────────────────────────────────────
    if not args.skip_extract:
        print(f"=== extraction {FEAT_KEY} (par sujet, stades atomiques) ===")
        Parallel(n_jobs=args.n_jobs)(
            delayed(extract_subject)(args.deriv_path, args.save_path, sub_id, args.overwrite_feat)
            for sub_id in SUBJECT_IDS
        )
    else:
        print("=== extraction sautee (--skip-extract) ===")

    # ── PARTIE 2 : classification ────────────────────────────────────────────
    if args.force_n_trials is not None:
        n_trials = args.force_n_trials
        print(f"n_trials = {n_trials} (FORCE)")
    else:
        n_trials = compute_n_trials(args.save_path, FEAT_KEY, states)
        print(f"n_trials = {n_trials} (min sur {FEAT_KEY}, etats {states})")

    results_dir = args.save_path / "results_spec_entropy_osc"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for state in states:
        print(f"\n=== classification {FEAT_KEY} x {state} ===")
        try:
            res = classify_vector(
                args.save_path, FEAT_KEY, state, n_trials,
                args.n_bootstraps, args.n_perm, args.normalize, args.n_jobs,
            )
        except Exception:
            print(f"ERROR {FEAT_KEY} {state}\n{traceback.format_exc()}")
            continue
        if res is None:
            continue

        out = results_dir / f"{FEAT_KEY}_{state}.npz"
        np.savez_compressed(out, **res)
        print(f"  -> {out}")

        ch_names = res["ch_names"].tolist()
        for e, ch in enumerate(ch_names):
            rows.append(dict(
                key=FEAT_KEY, state=state, electrode=ch,
                acc_mean=float(res["acc_mean"][e]),
                acc_std=float(res["acc_std"][e]),
                pval=float(res["pvals"][e]) if "pvals" in res else np.nan,
                pval_maxstat=float(res["pvals_maxstat"][e]) if "pvals_maxstat" in res else np.nan,
                n_trials=int(res["n_trials"]),
                n_subjects=int(res["n_subjects"]),
            ))
        # apercu console : meilleure electrode
        am = res["acc_mean"]
        best = int(np.argmax(am))
        pv_ms = res["pvals_maxstat"][best] if "pvals_maxstat" in res else np.nan
        print(f"  best: {ch_names[best]} acc={am[best]*100:.2f}% "
              f"p_maxstat={pv_ms:.4f}  (n_sig maxstat<0.05: "
              f"{int(np.sum(res['pvals_maxstat'] < 0.05)) if 'pvals_maxstat' in res else 0}/19)")

    if rows:
        csv = results_dir / f"{FEAT_KEY}_summary.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\nCSV : {csv}")

    m, s = divmod(int(time() - t0), 60)
    print(f"\ntotal : {m}m{s:02d}s")
