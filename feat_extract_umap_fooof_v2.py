"""Extract spectral, connectivity, and complexity features from preprocessed EEG.

Lit depuis derivatives/preprocessed-ica/ (sortie de preprocess_subject.py).
Remplace compute_psd.py, compute_cov.py, compute_cosp.py du repo Arthur.

Architecture
------------
Chaque enregistrement est segmenté UNE SEULE FOIS en epochs non-chevauchants
de 30s groupés par stade atomique (S1, S2, S3, S4, REM), lus depuis
_events.tsv. Les features (PSD brute, PSD oscillatoire FOOOF, exposant
aperiodic, covariance temporelle, cospectrum) sont calculées une fois par
groupe atomique et cachées sur disque.

Les états de classification (S2, SWS, REM, NREM) et les états UMAP (S1, S2,
SWS, REM) sont obtenus par concaténation des tableaux atomiques cachés —
sans relecture des données brutes ni recalcul (cf CLASSIFICATION_GROUPS /
UMAP_GROUPS dans config.py).

Notes
-----
- Données en entrée : derivatives/preprocessed-ica/, 19 canaux EEG, 250Hz,
  average reference, ICA appliqué. Différent du pipeline Arthur (référence nez,
  1000Hz, pas d'ICA) -> cov/cosp non directement comparables.
- Covariances() utilise l'estimateur SCM par défaut (pas de shrinkage),
  cohérent avec le pipeline original d'Arthur. Ne pas changer en "oas"/"lwf"
  sans documenter la déviation.
- Fichiers combinés .npz avec dtype=object (n_epochs variable par sujet) :
  charger avec np.load(path, allow_pickle=True).
- FOOOF (Donoghue et al. 2020, specparam) pour la séparation aperiodic/oscillatoire.
- Entropie/complexité (permutation entropy, Higuchi FD, etc.) planifiées via
  antropy (R. Vallat, co-auteur du dataset chapitre 1 de la thèse,
  https://github.com/raphaelvallat/antropy) — non encore implémentées.

Usage :
    python feat_extract_umap_fooof.py \\
        --deriv-path /path/to/derivatives/preprocessed-ica \\
        --save-path  /path/to/dream_features \\
        --n-jobs     -1

Author: based on Dehgan et al. sleep repo, modernised.
"""

import argparse
import traceback
from itertools import product
from pathlib import Path
from time import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import umap
from fooof import FOOOFGroup
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances, CospCovariances
from sklearn.preprocessing import StandardScaler

from config import (
    SFREQ_PREPROC, PER_BLACKLIST_STR, JBE_SUBJECTS_STR,
    N_SAMPLES, N_EEG, CH_NAMES,
    WINDOW, OVERLAP, FREQ_DICT, FOOOF_FREQ_RANGE,
    ATOMIC_STAGES, STAGE_LABEL_TO_ATOMIC,
    CLASSIFICATION_GROUPS, STATE_LIST,
    UMAP_GROUPS, UMAP_STATES, UMAP_COLORS,
    FEATURE_KEYS, SUBJECT_IDS,
)

SF = int(SFREQ_PREPROC)  # 250 Hz après décimation dans preprocess_subject.py


# ─── CLI ──────────────────────────────────────────────────────────────────────
# Défini juste après les imports/constantes : argparse ne dépend que de la
# stdlib + config, pas des fonctions ci-dessous. Contrat d'entrée visible
# d'un coup d'oeil avant l'implémentation.

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine du derivative preprocessed-ica "
                        "(ex: /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica)")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Dossier de sortie des features "
                        "(ex: /home/alouis/scratch/dream_features)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs joblib (défaut: tous les CPUs)")
    return p.parse_args()


# ─── path helpers ─────────────────────────────────────────────────────────────
# Format BIDS derivatives avec entité processing='clean' (proc-clean)
# produite par mne_bids.write_raw_bids(..., processing='clean').

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


# ─── epoch loading (single pass per subject) ──────────────────────────────────

def load_epochs_by_atomic_stage(
    deriv_path: Path, sub_id: str
) -> dict[str, np.ndarray]:
    """
    Lit le raw preprocessé + _events.tsv une seule fois.
    Coupe des epochs non-chevauchants de 30s, groupés par stade atomique.

    Returns dict[atomic_stage] -> (n_epochs, 19, 7500) à 250Hz.
    """
    raw = mne.io.read_raw_brainvision(
        _vhdr(deriv_path, sub_id), preload=True, verbose=False
    )
    raw.pick(CH_NAMES[:N_EEG])  # sélection par nom (robuste à l'ordre des canaux)
    n_total = raw.n_times

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"

    df = pd.read_csv(_events(deriv_path, sub_id), sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = (df[df["stage"].isin(STAGE_LABEL_TO_ATOMIC)]
          .sort_values("sample")
          .reset_index(drop=True))

    epochs: dict[str, list[np.ndarray]] = {s: [] for s in ATOMIC_STAGES}

    i = 0
    while i + 29 < len(df):
        block   = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages  = block["stage"].values

        # 30 secondes consécutives du même stade (1 annotation/seconde à SF Hz)
        if not (np.all(samples == samples[0] + np.arange(30) * SF) and
                np.all(stages == stages[0])):
            i += 1
            continue

        end = int(samples[0]) + N_SAMPLES
        if end > n_total:
            i += 1
            continue

        epoch = raw.get_data(start=int(samples[0]), stop=end)  # (19, 7500)
        epochs[STAGE_LABEL_TO_ATOMIC[stages[0]]].append(epoch)
        i += 30  # non-chevauchant

    return {s: np.stack(e) for s, e in epochs.items() if e}


# ─── feature computation ──────────────────────────────────────────────────────

def compute_psd_spectrum(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_epochs, 19, 7500) -> psds (n_epochs, 19, n_freqs), freqs (n_freqs,).

    Un seul appel Welch sur le spectre complet 1-45Hz. Toutes les features
    spectrales (bandes brutes, bandes oscillatoires FOOOF) en sont dérivées
    sans second appel.
    """
    return mne.time_frequency.psd_array_welch(
        data,
        sfreq=SF,
        fmin=FOOOF_FREQ_RANGE[0],
        fmax=FOOOF_FREQ_RANGE[1],
        n_fft=WINDOW,
        n_overlap=OVERLAP,
        n_per_seg=WINDOW,
        window="hann",
        verbose=False,
    )


def band_power(
    spectrum: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float
) -> np.ndarray:
    """(n_epochs, 19, n_freqs) -> (n_epochs, 19) moyenne sur [fmin, fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return spectrum[..., mask].mean(axis=-1)


def fit_fooof(
    psds: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit FOOOF (mode aperiodic fixe) sur chaque spectre (epoch, canal).

    FOOOF / specparam — Donoghue et al. 2020, Nature Neuroscience.
    aperiodic_mode="fixed" : pas de knee, adapté à la plage 1-45Hz
    (à réévaluer si les résultats delta SWS semblent aberrants).

    n_jobs=1 : le parallélisme est géré au niveau sujet par joblib en amont.

    Returns
    -------
    exponent  : (n_epochs, 19)         pente aperiodic (exposant 1/f)
    flattened : (n_epochs, 19, n_freqs) log10(psd) - fit_aperiodic (oscillatoire)
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs)

    fg = FOOOFGroup(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)

    aperiodic = fg.get_params("aperiodic_params")  # (n, 2) [offset, exponent]
    exponent  = aperiodic[:, 1].reshape(n_epochs, n_ch)

    offsets   = aperiodic[:, 0:1]
    exponents = aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]
    flattened_log = np.log10(flat_psds) - ap_fit_log

    return exponent, flattened_log.reshape(n_epochs, n_ch, n_freqs)


def compute_cov(data: np.ndarray) -> np.ndarray:
    """(n_epochs, 19, 7500) -> (n_epochs, 19, 19).

    Estimateur SCM (Sample Covariance Matrix) par défaut, cohérent avec
    le pipeline original d'Arthur. Ne pas changer en OAS/LWF sans documenter.
    """
    return Covariances().fit_transform(data)


def compute_cosp(
    data: np.ndarray, fmin: float, fmax: float
) -> np.ndarray:
    """(n_epochs, 19, 7500) -> (n_epochs, 19, 19) cospectrum moyen sur la bande.

    Mêmes paramètres Welch que compute_psd_spectrum (WINDOW, OVERLAP)
    pour cohérence avec la thèse §1.2.6.
    """
    mat = CospCovariances(
        window=WINDOW, overlap=0.0, fmin=fmin, fmax=fmax, fs=SF
    ).fit_transform(data)
    return mat.mean(axis=-1) if mat.ndim == 4 else mat


def compute_all_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """Un groupe d'epochs -> toutes les features en un seul passage Welch/cov/cosp."""
    psds, freqs    = compute_psd_spectrum(data)
    exponent, flat = fit_fooof(psds, freqs)

    feats: dict[str, np.ndarray] = {
        "aperiodic": exponent,
        "cov":       compute_cov(data),
    }
    for fname, (fmin, fmax) in FREQ_DICT.items():
        feats[f"psd_{fname}"]     = band_power(psds,  freqs, fmin, fmax)
        feats[f"psd_osc_{fname}"] = band_power(flat,  freqs, fmin, fmax)
        feats[f"cosp_{fname}"]    = compute_cosp(data, fmin, fmax)
    return feats


# ─── per-subject pipeline ─────────────────────────────────────────────────────

def process_subject(
    deriv_path: Path, save_path: Path, sub_id: str
) -> None:
    if not _vhdr(deriv_path, sub_id).exists():
        print(f"sub-{sub_id}: derivative not found, skipping")
        return

    try:
        atomic_epochs = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        print(f"sub-{sub_id}: ERROR loading\n{traceback.format_exc()}")
        return

    for stage, data in atomic_epochs.items():
        print(f"  sub-{sub_id} {stage}: {data.shape[0]} epochs")

        if all(
            (save_path / k / f"{k}_s{sub_id}_{stage}.npz").exists()
            for k in FEATURE_KEYS
        ):
            print(f"  sub-{sub_id} {stage}: already cached, skipping")
            continue

        try:
            feats = compute_all_features(data)
        except Exception:
            print(f"sub-{sub_id} {stage}: ERROR features\n{traceback.format_exc()}")
            continue

        for key, arr in feats.items():
            out = save_path / key / f"{key}_s{sub_id}_{stage}.npz"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=arr)

    print(f"sub-{sub_id}: done")


# ─── combine: atomique par sujet -> états de classification (tous sujets) ─────

def _load_atomic(
    save_path: Path, key: str, sub_id: str, stage: str
) -> np.ndarray | None:
    f = save_path / key / f"{key}_s{sub_id}_{stage}.npz"
    return np.load(f)["data"] if f.exists() else None


def combine_classification_state(
    save_path: Path, key: str, state: str
) -> None:
    """Concatène les tableaux atomiques par CLASSIFICATION_GROUPS[state], empile les sujets.

    Fichier combined : dtype=object (n_epochs variable par sujet).
    Charger avec np.load(path, allow_pickle=True).
    """
    out = save_path / key / f"{key}_{state}.npz"
    if out.exists():
        return

    stages = CLASSIFICATION_GROUPS[state]
    arrays = []
    for sub_id in SUBJECT_IDS:
        parts = [
            a for s in stages
            if (a := _load_atomic(save_path, key, sub_id, s)) is not None
        ]
        if parts:
            arrays.append(np.concatenate(parts, axis=0))

    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


# ─── UMAP ─────────────────────────────────────────────────────────────────────

def _upper_tri(arr: np.ndarray) -> np.ndarray:
    """(n, p, p) -> (n, p*(p+1)/2) triangle supérieur."""
    idx = np.triu_indices(arr.shape[-1])
    return arr[..., idx[0], idx[1]].reshape(len(arr), -1)


UMAP_FEATURE_GROUPS = {
    "psd":       [f"psd_{b}"     for b in FREQ_DICT],
    "psd_osc":   [f"psd_osc_{b}" for b in FREQ_DICT],
    "cov":       ["cov"],
    "cosp":      [f"cosp_{b}"    for b in FREQ_DICT],
    "aperiodic": ["aperiodic"],
}


def build_umap_vectors(
    save_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Construit les vecteurs UMAP (6 groupes incl. 'all') depuis les .npz atomiques cachés."""
    vectors: dict[str, list[np.ndarray]] = {g: [] for g in UMAP_FEATURE_GROUPS}
    labels: list[str] = []

    for sub_id in SUBJECT_IDS:
        for state, stages in UMAP_GROUPS.items():
            per_key: dict[str, np.ndarray] = {}
            n_epochs = None
            ok = True

            for key in FEATURE_KEYS:
                parts = [
                    a for s in stages
                    if (a := _load_atomic(save_path, key, sub_id, s)) is not None
                ]
                if not parts:
                    ok = False
                    break
                arr = np.concatenate(parts, axis=0)
                if key == "cov" or key.startswith("cosp"):
                    arr = _upper_tri(arr)
                per_key[key] = arr
                n_epochs = arr.shape[0]

            if not ok:
                continue

            for group, keys in UMAP_FEATURE_GROUPS.items():
                vectors[group].append(
                    np.concatenate([per_key[k] for k in keys], axis=1)
                )
            labels.extend([state] * n_epochs)

    out = {g: np.concatenate(v, axis=0) for g, v in vectors.items()}
    out["all"] = np.concatenate([out[g] for g in UMAP_FEATURE_GROUPS], axis=1)
    return out, np.array(labels)


def build_umap_vectors_cached(
    save_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cache les vecteurs UMAP pour éviter de tout recalculer si le plot crashe."""
    cache = save_path / "umap_vectors.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        vectors = {k: d[k] for k in d.files if k != "labels"}
        return vectors, d["labels"]

    vectors, labels = build_umap_vectors(save_path)
    np.savez_compressed(cache, labels=labels, **vectors)
    return vectors, labels


def plot_umaps(
    vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    titles = {
        "psd":       "PSD (bandes brutes)",
        "psd_osc":   "PSD oscillatoire (corrigée 1/f)",
        "cov":       "Covariance",
        "cosp":      "Cospectrum (toutes bandes)",
        "aperiodic": "Exposant apériodique",
        "all":       "Toutes features combinées",
    }

    for ax, (fname, title) in zip(axes.flatten(), titles.items()):
        print(f"  UMAP: {fname}")
        X   = StandardScaler().fit_transform(vectors[fname])
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(X)

        for state in UMAP_STATES:
            mask = labels == state
            if mask.any():
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=UMAP_COLORS[state], s=3, alpha=0.4, rasterized=True,
                )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(
            handles=[mpatches.Patch(color=UMAP_COLORS[s], label=s) for s in UMAP_STATES],
            markerscale=3,
        )

    fig.suptitle("UMAP — séparabilité des stades de sommeil par type de feature", fontsize=15)
    plt.tight_layout()
    out = save_path / "umap_sleep_stages.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args       = parse_args()
    deriv_path = args.deriv_path
    save_path  = args.save_path
    n_jobs     = args.n_jobs

    t0 = time()

    print("=== extraction features par sujet (stades atomiques) ===")
    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(deriv_path, save_path, sub_id)
        for sub_id in SUBJECT_IDS
    )

    print("=== combinaison en états de classification ===")
    Parallel(n_jobs=n_jobs)(
        delayed(combine_classification_state)(save_path, key, state)
        for key, state in product(FEATURE_KEYS, STATE_LIST)
    )

    print("=== UMAP ===")
    vectors, labels = build_umap_vectors_cached(save_path)
    plot_umaps(vectors, labels, save_path)

    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")
