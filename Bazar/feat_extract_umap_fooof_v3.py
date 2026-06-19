"""Extract spectral, connectivity, and complexity features from preprocessed EEG.

Lit depuis derivatives/preprocessed-ica/ (sortie de preprocess_subject_v2.py).
Remplace compute_psd.py, compute_cov.py, compute_cosp.py du repo Arthur.

Architecture
------------
Chaque enregistrement est segmenté UNE SEULE FOIS en epochs non-chevauchants
de 30s groupés par stade atomique (S1, S2, S3, S4, REM), lus depuis
_events.tsv. Les features (PSD brute, PSD oscillatoire FOOOF, exposant
aperiodic, covariance temporelle, cospectrum) sont calculées une fois par
groupe atomique et cachées sur disque.

Les états de classification (S2, SWS, REM, NREM) sont obtenus par
concaténation des tableaux atomiques cachés — sans relecture des données
brutes ni recalcul (cf CLASSIFICATION_GROUPS dans config.py).

La visualisation UMAP est séparée dans visualize_umap.py, qui lit les mêmes
.npz atomiques.

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
    python feat_extract.py \\
        --deriv-path /path/to/derivatives/preprocessed-ica \\
        --save-path  /path/to/dream_features \\
        --n-jobs     $SLURM_CPUS_PER_TASK \\
        --overwrite  # optionnel : écrase les .npz existants

"""

import argparse
import traceback
from itertools import product
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import mne
from specparam import SpectralGroupModel
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances, CoSpectra

from config import (
    SFREQ_PREPROC, PER_BLACKLIST_STR, JBE_SUBJECTS_STR,
    N_SAMPLES, N_EEG, CH_NAMES,
    WINDOW, OVERLAP, FREQ_DICT, FOOOF_FREQ_RANGE,
    ATOMIC_STAGES, STAGE_LABEL_TO_ATOMIC,
    CLASSIFICATION_GROUPS, STATE_LIST,
    FEATURE_KEYS, SUBJECT_IDS,
)
from utils import load_atomic

SF = int(SFREQ_PREPROC)  # 250 Hz après décimation dans le prepro


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine du derivative preprocessed-ica "
                        "(ex: /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica)")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Dossier de sortie des features "
                        "(ex: /home/alouis/scratch/dream_features)")
    p.add_argument("--n-jobs", type=int, default=1, 
                   help="Parallel jobs joblib (défaut: 1 CPU)")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Écrase les .npz existants (utile après changement de params)")
    return p.parse_args()


# ─── path helpers ─────────────────────────────────────────────────────────────
#chemins vers les fichiers preprocessed (proc-clean) 
#produits par preprocess_subject_v2.py

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
    raw.pick(CH_NAMES[:N_EEG])  # selection par nom
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

        if not (np.all(samples == samples[0] + np.arange(30) * SF) and 
                np.all(stages == stages[0])):
            #on verifie si  les 30 annotations sont espacées 
            #exactement de 250 samples (1s à 250Hz), sans trou ni saut
            #et que toutes les 30 secondes appartiennent au même stade
            i += 1
            continue
        
        end = int(samples[0]) + N_SAMPLES

        # verifie que l'epoch ne depasse pas la fin du fichier on sait jamais
        if end > n_total:
            raise ValueError(
                f"sub-{sub_id}: epoch dépasse la fin du fichier "
                f"(end={end}, n_total={n_total})"
            )
        epoch = raw.get_data(start=int(samples[0]), stop=end)  # (19, 7500)
        epochs[STAGE_LABEL_TO_ATOMIC[stages[0]]].append(epoch)
        i += 30  # un pas de 30s 

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
    #masque booleen sur frequences ex : 1 2 3 4 true et le reste false pr prem bande
    return spectrum[..., mask].mean(axis=-1)
    #extrait la puissance moyenne sur une bande de frequences

def fit_fooof(
    psds: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit FOOOF (mode aperiodic fixe) sur chaque spectre (epoch, canal).

    specparam (ex-FOOOF) — Donoghue et al. 2020, Nature Neuroscience.
    aperiodic_mode="fixed" : pas de knee, adapté à la plage 1-45Hz
    (à réévaluer si les résultats delta SWS semblent aberrants).

    n_jobs=1 : le parallélisme est géré au niveau sujet par joblib en amont.

    Returns:
    exponent  : (n_epochs, 19)         pente aperiodic (exposant 1/f)
    flattened : (n_epochs, 19, n_freqs) log10(psd) - fit_aperiodic (oscillatoire)
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs) #specparam attend un array 2D
    #la résolution freq est SF/WINDOW = 1Hz. Donc sur 1-45Hz => nfreq = 45

    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)
    #parallelisme est géré en amont => n_jobs =1

    aperiodic = fg.get_params("aperiodic_params")
    # (n_epochs*19, 2) : col0=offset , col1=exposant
    #offset = coordonées a l'origine et exponent = pente (>0)
    exponent  = aperiodic[:, 1].reshape(n_epochs, n_ch)

    offsets   = aperiodic[:, 0:1]
    exponents = aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]
    flattened_log = np.log10(flat_psds) - ap_fit_log
    #flatten log = residu oscillatoire

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

    Mêmes paramètres Welch que compute_psd_spectrum (WINDOW) pour cohérence
    avec la thèse §1.2.6. overlap=0.01 (quasi no-overlap) : pyriemann 0.11
    refuse overlap=0.0 strictement (ValueError). 0.01 sur 250 samples = 2-3
    samples de chevauchement, négligeable, cohérent avec l'esprit Arthur.
    """
    mat = CoSpectra(
        window=WINDOW, overlap=0.01, fmin=fmin, fmax=fmax, fs=SF
    ).fit_transform(data)
    return mat.mean(axis=-1) if mat.ndim == 4 else mat
    # pyriemann retourne (n_epochs, 19, 19, n_freqs) ou (n_epochs, 19, 19)
    # selon la version -> moyenne sur l'axe fréquences si 4D
    #a tester quand cluster plus down


def compute_all_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """Un groupe d'epochs -> toutes les features en un seul passage.

    Un seul appel Welch + FOOOF, toutes les features spectrales en sont dérivées.
    CoSpectra appelé une fois par bande.

    Returns dict avec clés : aperiodic, cov, psd_{band}, psd_osc_{band}, cosp_{band}.
    """
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
    deriv_path: Path, save_path: Path, sub_id: str, overwrite: bool = False
) -> None:
    if not _vhdr(deriv_path, sub_id).exists():
        print(f"sub-{sub_id}: derivative not found, skipping")
        return
    #Verifie que le fichier preprocessé existe sur disque

    try:
        atomic_epochs = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        print(f"sub-{sub_id}: ERROR loading\n{traceback.format_exc()}")
        return
    #charge le raw et decoupe en epochs

    for stage, data in atomic_epochs.items():
        print(f"  sub-{sub_id} {stage}: {data.shape[0]} epochs")

        # skip si tous les .npz de ce sujet/stade existent -> reprise apres crash cluster
        if not overwrite and all(
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
        #calcul des features

        for key, arr in feats.items():
            out = save_path / key / f"{key}_s{sub_id}_{stage}.npz"
            # double check par feature : protege contre un crash entre deux saves
            if not out.exists() or overwrite:
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=arr)
                #sauvegarde l'array compressé
                
    print(f"sub-{sub_id}: done")


# ─── combine: atomique par sujet -> états de classification (tous sujets) ─────

def combine_classification_state(
    save_path: Path, key: str, state: str, overwrite: bool = False
) -> None:
    """Concatène les tableaux atomiques par CLASSIFICATION_GROUPS[state], empile les sujets.

    Fichier combined : dtype=object (n_epochs variable par sujet).
    Charger avec np.load(path, allow_pickle=True).
    """
    out = save_path / key / f"{key}_{state}.npz"
    if out.exists() and not overwrite:
        return

    stages = CLASSIFICATION_GROUPS[state]
    arrays = []
    for sub_id in SUBJECT_IDS:
        parts = [
            a for s in stages
            if (a := load_atomic(save_path, key, sub_id, s)) is not None
        ]
        if parts:
            arrays.append(np.concatenate(parts, axis=0))

    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args       = parse_args()
    deriv_path = args.deriv_path
    save_path  = args.save_path
    n_jobs     = args.n_jobs
    overwrite  = args.overwrite

    t0 = time()

    print("=== extraction features par sujet (stades atomiques) ===")
    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(deriv_path, save_path, sub_id, overwrite)
        for sub_id in SUBJECT_IDS
    )

    print("=== combinaison en états de classification ===")
    Parallel(n_jobs=n_jobs)(
        delayed(combine_classification_state)(save_path, key, state, overwrite)
        for key, state in product(FEATURE_KEYS, STATE_LIST)
    )

    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")
    print("Lancer visualize_umap.py --save-path <save_path> pour le UMAP.")

