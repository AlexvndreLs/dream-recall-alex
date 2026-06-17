"""Extract PSD, covariance, and cospectrum features from BIDS BrainVision EEG.

Replaces compute_psd.py, compute_cov.py, compute_cosp.py.
Reads 30s epochs from _events.tsv (BIDS), scorer=per by default (jbe fallback).
Saves per-subject .npz, then combines into one .npz per (state, feature, [freq]).

Author: based on Dehgan et al. sleep repo, modernised.
"""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances, CospCovariances

# ─── paths ────────────────────────────────────────────────────────────────────
BIDS_PATH = Path("/home/alouis/scratch/dream_bids")
SAVE_PATH = Path("/home/alouis/scratch/dream_features")

# ─── params ───────────────────────────────────────────────────────────────────
SF = 1000
EPOCH_DURATION = 30.0
N_SAMPLES = int(SF * EPOCH_DURATION)  # 30 000

WINDOW = 1000   # Welch window in samples (1s Hanning, no overlap) — thesis §1.2.5
OVERLAP = 0

FREQ_DICT = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (11, 16),
    "beta":  (17, 35),
}

N_EEG = 19  # first 19 channels are EEG (Ch1-Ch19 in vhdr)

STAGE_TO_STATES = {
    "Sleep stage S1": ["NREM"],
    "Sleep stage S2": ["S2", "NREM"],
    "Sleep stage S3": ["SWS", "NREM"],
    "Sleep stage S4": ["SWS", "NREM"],
    "Sleep stage R":  ["REM"],
}
STATE_LIST = ["S2", "SWS", "REM", "NREM"]

SUBJECT_IDS = [f"{i:02d}" for i in range(1, 39)]

PER_BLACKLIST = {"19"}
JBE_SUBJECTS = {"01","02","04","06","10","11","14","16","18","19",
                 "23","25","26","29","32","33","35","37"}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _vhdr_path(sub_id: str) -> Path:
    return BIDS_PATH / f"sub-{sub_id}" / "eeg" / f"sub-{sub_id}_task-sleep_eeg.vhdr"


def _events_path(sub_id: str) -> Path:
    return BIDS_PATH / f"sub-{sub_id}" / "eeg" / f"sub-{sub_id}_task-sleep_events.tsv"


def _choose_scorer(sub_id: str) -> str:
    if sub_id not in PER_BLACKLIST:
        return "per"
    if sub_id in JBE_SUBJECTS:
        return "jbe"
    raise ValueError(f"sub-{sub_id}: no valid scorer available")


def load_epochs_by_state(sub_id: str) -> dict[str, np.ndarray]:
    """
    Load raw (19 EEG channels) and segment into non-overlapping 30s epochs
    per sleep state, reading stage labels from _events.tsv.

    Returns
    -------
    dict[state] -> np.ndarray of shape (n_epochs, 19, 30000)
    """
    raw = mne.io.read_raw_brainvision(_vhdr_path(sub_id), preload=True, verbose=False)
    raw.pick(raw.ch_names[:N_EEG])
    n_total_samples = raw.n_times

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"

    df = pd.read_csv(_events_path(sub_id), sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = df[df["stage"].isin(STAGE_TO_STATES)]
    df = df.sort_values("sample").reset_index(drop=True)

    epochs_by_state: dict[str, list[np.ndarray]] = {s: [] for s in STATE_LIST}

    i = 0
    while i < len(df):
        if i + 29 >= len(df):
            break
        block = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages = block["stage"].values

        is_consecutive = np.all(samples == samples[0] + np.arange(30) * SF)
        is_same_stage = np.all(stages == stages[0])

        if not (is_consecutive and is_same_stage):
            i += 1
            continue

        onset_sample = int(samples[0])
        end_sample = onset_sample + N_SAMPLES
        if end_sample > n_total_samples:
            i += 1
            continue

        epoch = raw.get_data(start=onset_sample, stop=end_sample)  # (19, 30000)
        for state in STAGE_TO_STATES[stages[0]]:
            epochs_by_state[state].append(epoch)

        i += 30  # non-overlapping

    return {
        state: np.stack(epochs, axis=0)  # (n_epochs, 19, 30000)
        for state, epochs in epochs_by_state.items()
        if len(epochs) > 0
    }


# ─── feature computation ──────────────────────────────────────────────────────

def compute_psd_subject(data: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Mean PSD per epoch per channel in [fmin, fmax]. Returns (n_epochs, 19)."""
    psds, _ = mne.time_frequency.psd_array_welch(
        data,
        sfreq=SF,
        fmin=fmin,
        fmax=fmax,
        n_fft=WINDOW,
        n_overlap=OVERLAP,
        n_per_seg=WINDOW,
        window="hann",
        verbose=False,
    )
    return psds.mean(axis=-1)  # (n_epochs, 19)


def compute_cov_subject(data: np.ndarray) -> np.ndarray:
    """Time covariance matrix per epoch. Returns (n_epochs, 19, 19)."""
    return Covariances(estimator="oas").fit_transform(data)


def compute_cosp_subject(data: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Cospectrum matrix per epoch averaged over band. Returns (n_epochs, 19, 19)."""
    cosp = CospCovariances(
        window=WINDOW,
        overlap=0.0,
        fmin=fmin,
        fmax=fmax,
        fs=SF,
    )
    mat = cosp.fit_transform(data)
    if mat.ndim == 4:
        mat = mat.mean(axis=-1)
    return mat


# ─── per-subject pipeline ─────────────────────────────────────────────────────

def process_subject(sub_id: str) -> None:
    if not _vhdr_path(sub_id).exists():
        print(f"sub-{sub_id}: not found, skipping")
        return

    print(f"sub-{sub_id}: loading")
    try:
        epochs_by_state = load_epochs_by_state(sub_id)
    except Exception as e:
        print(f"sub-{sub_id}: ERROR — {e}")
        return

    for state, data in epochs_by_state.items():
        print(f"  sub-{sub_id} {state}: {data.shape[0]} epochs")

        for fname, (fmin, fmax) in FREQ_DICT.items():
            out = SAVE_PATH / "psd" / f"psd_s{sub_id}_{state}_{fname}.npz"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=compute_psd_subject(data, fmin, fmax))

        out = SAVE_PATH / "cov" / f"cov_s{sub_id}_{state}.npz"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, data=compute_cov_subject(data))

        for fname, (fmin, fmax) in FREQ_DICT.items():
            out = SAVE_PATH / "cosp" / f"cosp_s{sub_id}_{state}_{fname}.npz"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=compute_cosp_subject(data, fmin, fmax))

    print(f"sub-{sub_id}: done")


# ─── combine across subjects ──────────────────────────────────────────────────

def combine_psd(state: str, freq: str) -> None:
    out = SAVE_PATH / "psd" / f"psd_{state}_{freq}.npz"
    if out.exists():
        return
    arrays = [
        np.load(SAVE_PATH / "psd" / f"psd_s{s}_{state}_{freq}.npz")["data"]
        for s in SUBJECT_IDS
        if (SAVE_PATH / "psd" / f"psd_s{s}_{state}_{freq}.npz").exists()
    ]
    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


def combine_cov(state: str) -> None:
    out = SAVE_PATH / "cov" / f"cov_{state}.npz"
    if out.exists():
        return
    arrays = [
        np.load(SAVE_PATH / "cov" / f"cov_s{s}_{state}.npz")["data"]
        for s in SUBJECT_IDS
        if (SAVE_PATH / "cov" / f"cov_s{s}_{state}.npz").exists()
    ]
    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


def combine_cosp(state: str, freq: str) -> None:
    out = SAVE_PATH / "cosp" / f"cosp_{state}_{freq}.npz"
    if out.exists():
        return
    arrays = [
        np.load(SAVE_PATH / "cosp" / f"cosp_s{s}_{state}_{freq}.npz")["data"]
        for s in SUBJECT_IDS
        if (SAVE_PATH / "cosp" / f"cosp_s{s}_{state}_{freq}.npz").exists()
    ]
    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from time import time
    t0 = time()

    print("=== per-subject feature extraction ===")
    Parallel(n_jobs=-1)(
        delayed(process_subject)(sub_id) for sub_id in SUBJECT_IDS
    )

    print("=== combining subjects ===")
    Parallel(n_jobs=-1)(
        delayed(combine_psd)(state, freq)
        for state, freq in product(STATE_LIST, FREQ_DICT)
    )
    Parallel(n_jobs=-1)(
        delayed(combine_cov)(state) for state in STATE_LIST
    )
    Parallel(n_jobs=-1)(
        delayed(combine_cosp)(state, freq)
        for state, freq in product(STATE_LIST, FREQ_DICT)
    )

    print("=== UMAP ===")
    vectors, labels = build_umap_data()
    plot_umaps(vectors, labels)

    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")

# ─── UMAP ─────────────────────────────────────────────────────────────────────

UMAP_STATES = ["S1", "S2", "SWS", "REM"]
UMAP_COLORS = {"S1": "#4C72B0", "S2": "#DD8452", "SWS": "#55A868", "REM": "#C44E52"}


def _load_all_epochs_umap(sub_id: str) -> dict[str, np.ndarray]:
    """Load raw epochs per atomic state for a subject (for UMAP only)."""
    raw = mne.io.read_raw_brainvision(_vhdr_path(sub_id), preload=True, verbose=False)
    raw.pick(raw.ch_names[:N_EEG])
    n_total_samples = raw.n_times

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"

    stage_to_umap = {
        "Sleep stage S1": "S1",
        "Sleep stage S2": "S2",
        "Sleep stage S3": "SWS",
        "Sleep stage S4": "SWS",
        "Sleep stage R":  "REM",
    }

    df = pd.read_csv(_events_path(sub_id), sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = df[df["stage"].isin(stage_to_umap)]
    df = df.sort_values("sample").reset_index(drop=True)

    epochs_by_state: dict[str, list[np.ndarray]] = {s: [] for s in UMAP_STATES}

    i = 0
    while i < len(df):
        if i + 29 >= len(df):
            break
        block = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages = block["stage"].values
        if not (np.all(samples == samples[0] + np.arange(30) * SF) and
                np.all(stages == stages[0])):
            i += 1
            continue
        onset_sample = int(samples[0])
        end_sample = onset_sample + N_SAMPLES
        if end_sample > n_total_samples:
            i += 1
            continue
        umap_state = stage_to_umap[stages[0]]
        epochs_by_state[umap_state].append(
            raw.get_data(start=onset_sample, stop=end_sample)
        )
        i += 30

    return {s: np.stack(e, axis=0) for s, e in epochs_by_state.items() if len(e) > 0}


def _flatten_matrix(arr: np.ndarray) -> np.ndarray:
    """Upper triangle of symmetric matrix -> 1D per epoch."""
    n = arr.shape[-1]
    idx = np.triu_indices(n)
    return arr[..., idx[0], idx[1]].reshape(len(arr), -1)


def build_umap_data() -> tuple[dict, np.ndarray]:
    print("=== loading epochs for UMAP ===")
    psd_vecs, cov_vecs, cosp_vecs, labels = [], [], [], []

    for sub_id in SUBJECT_IDS:
        if not _vhdr_path(sub_id).exists():
            continue
        try:
            state_data = _load_all_epochs_umap(sub_id)
        except Exception as e:
            print(f"  sub-{sub_id}: {e}")
            continue

        for state, data in state_data.items():
            psd_vecs.append(np.concatenate(
                [compute_psd_subject(data, fmin, fmax)
                 for fmin, fmax in FREQ_DICT.values()], axis=1))
            cov_vecs.append(_flatten_matrix(compute_cov_subject(data)))
            cosp_vecs.append(np.concatenate(
                [_flatten_matrix(compute_cosp_subject(data, fmin, fmax))
                 for fmin, fmax in FREQ_DICT.values()], axis=1))
            labels.extend([state] * len(data))

    psd_all  = np.concatenate(psd_vecs,  axis=0)
    cov_all  = np.concatenate(cov_vecs,  axis=0)
    cosp_all = np.concatenate(cosp_vecs, axis=0)
    labels   = np.array(labels)

    vectors = {
        "psd":  psd_all,
        "cov":  cov_all,
        "cosp": cosp_all,
        "all":  np.concatenate([psd_all, cov_all, cosp_all], axis=1),
    }
    return vectors, labels


def plot_umaps(vectors: dict, labels: np.ndarray) -> None:
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.preprocessing import StandardScaler

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    feature_names = ["psd", "cov", "cosp", "all"]
    titles = ["PSD (all bands)", "Covariance", "Cospectrum (all bands)",
              "PSD + Cov + Cosp"]

    for ax, fname, title in zip(axes, feature_names, titles):
        print(f"  UMAP: {fname}")
        X = StandardScaler().fit_transform(vectors[fname])
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X)

        for state in UMAP_STATES:
            mask = labels == state
            if not mask.any():
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=UMAP_COLORS[state], s=3, alpha=0.4, rasterized=True,
            )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(handles=[
            mpatches.Patch(color=UMAP_COLORS[s], label=s) for s in UMAP_STATES
        ], markerscale=3)

    fig.suptitle("UMAP — sleep stage separability by feature type", fontsize=15)
    plt.tight_layout()
    out = SAVE_PATH / "umap_sleep_stages.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()
