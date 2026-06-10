"""Extract PSD, covariance, and cospectrum features from BIDS BrainVision EEG.

Replaces compute_psd.py, compute_cov.py, compute_cosp.py.
Reads 30s epochs from _events.tsv (BIDS), scorer=per by default (jbe fallback).
Saves per-subject .npz, then combines into one .npz per (state, feature, [freq]).
Generates 4 UMAP plots (psd / cov / cosp / all), coloured by sleep stage.

Author: based on Dehgan et al. sleep repo, modernised.
"""

import argparse
from itertools import product
from pathlib import Path
from time import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import umap
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances, CospCovariances
from sklearn.preprocessing import StandardScaler

from config import (
    SFREQ, PER_BLACKLIST_STR, JBE_SUBJECTS_STR, N_SAMPLES, N_EEG,
    WINDOW, OVERLAP, FREQ_DICT, STAGE_TO_STATES, STATE_LIST,
    STAGE_TO_UMAP, UMAP_STATES, UMAP_COLORS, SUBJECT_IDS,
)

# ─── constants ────────────────────────────────────────────────────────────────

SF = int(SFREQ)

# ─── path helpers ─────────────────────────────────────────────────────────────

# construct the file paths required to navigate the BIDS directory tree

def _vhdr(bids_path: Path, sub_id: str) -> Path:
    return bids_path / f"sub-{sub_id}" / "eeg" / f"sub-{sub_id}_task-sleep_eeg.vhdr"


def _events(bids_path: Path, sub_id: str) -> Path:
    return bids_path / f"sub-{sub_id}" / "eeg" / f"sub-{sub_id}_task-sleep_events.tsv"

# defaults to 'per' annotator
# if the subject is blacklisted (S19), falls back to 'jbe' annotation
def _choose_scorer(sub_id: str) -> str:
    if sub_id not in PER_BLACKLIST_STR:
        return "per"
    if sub_id in JBE_SUBJECTS_STR:
        return "jbe"
    raise ValueError(f"sub-{sub_id}: no valid scorer")


# ─── epoch loading ────────────────────────────────────────────────────────────

def _segment_epochs(
    raw: mne.io.BaseRaw,
    events_path: Path,
    scorer: str,
    stage_map: dict[str, str | list[str]],
) -> dict[str, list[np.ndarray]]:
    """
    Read _events.tsv and cut non-overlapping 30s epochs.
    stage_map values can be a str (atomic) or list[str] (aggregate).
    Returns dict[label -> list of (19, 30000) arrays].
    """
    prefix = f"{scorer}/"
    n_total = raw.n_times

    df = pd.read_csv(events_path, sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = df[df["stage"].isin(stage_map)].sort_values("sample").reset_index(drop=True)

    all_labels = set()
    for v in stage_map.values():
        if isinstance(v, list):
            all_labels.update(v)
        else:
            all_labels.add(v)
    result: dict[str, list[np.ndarray]] = {lbl: [] for lbl in all_labels}

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
            i += 1
            continue

        epoch  = raw.get_data(start=int(samples[0]), stop=end)  # (19, 30000)
        target = stage_map[stages[0]]
        for lbl in (target if isinstance(target, list) else [target]):
            result[lbl].append(epoch)
        i += 30

    return result


def load_epochs_by_state(
    bids_path: Path, sub_id: str
) -> dict[str, np.ndarray]:
    raw = mne.io.read_raw_brainvision(_vhdr(bids_path, sub_id), preload=True, verbose=False)
    raw.pick(raw.ch_names[:N_EEG])
    scorer  = _choose_scorer(sub_id)
    result  = _segment_epochs(raw, _events(bids_path, sub_id), scorer, STAGE_TO_STATES)
    return {s: np.stack(e) for s, e in result.items() if e}


def load_epochs_umap(
    bids_path: Path, sub_id: str
) -> dict[str, np.ndarray]:
    raw = mne.io.read_raw_brainvision(_vhdr(bids_path, sub_id), preload=True, verbose=False)
    raw.pick(raw.ch_names[:N_EEG])
    scorer  = _choose_scorer(sub_id)
    result  = _segment_epochs(raw, _events(bids_path, sub_id), scorer, STAGE_TO_UMAP)
    return {s: np.stack(e) for s, e in result.items() if e}


# ─── feature computation ──────────────────────────────────────────────────────

def compute_psd(data: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """(n_epochs, 19, 30000) -> (n_epochs, 19) mean PSD in band."""
    psds, _ = mne.time_frequency.psd_array_welch(
        data, sfreq=SF, fmin=fmin, fmax=fmax,
        n_fft=WINDOW, n_overlap=OVERLAP, n_per_seg=WINDOW,
        window="hann", verbose=False,
    )
    return psds.mean(axis=-1)


def compute_cov(data: np.ndarray) -> np.ndarray:
    """(n_epochs, 19, 30000) -> (n_epochs, 19, 19)."""
    return Covariances(estimator="oas").fit_transform(data)


def compute_cosp(data: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """(n_epochs, 19, 30000) -> (n_epochs, 19, 19) mean cospectrum in band."""
    mat = CospCovariances(window=WINDOW, overlap=0.0, fmin=fmin, fmax=fmax, fs=SF).fit_transform(data)
    return mat.mean(axis=-1) if mat.ndim == 4 else mat


# ─── per-subject pipeline ─────────────────────────────────────────────────────

def process_subject(bids_path: Path, save_path: Path, sub_id: str) -> None:
    if not _vhdr(bids_path, sub_id).exists():
        print(f"sub-{sub_id}: not found, skipping")
        return
    try:
        epochs_by_state = load_epochs_by_state(bids_path, sub_id)
    except Exception as e:
        print(f"sub-{sub_id}: ERROR — {e}")
        return

    for state, data in epochs_by_state.items():
        print(f"  sub-{sub_id} {state}: {data.shape[0]} epochs")

        for fname, (fmin, fmax) in FREQ_DICT.items():
            out = save_path / "psd" / f"psd_s{sub_id}_{state}_{fname}.npz"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=compute_psd(data, fmin, fmax))

        out = save_path / "cov" / f"cov_s{sub_id}_{state}.npz"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, data=compute_cov(data))

        for fname, (fmin, fmax) in FREQ_DICT.items():
            out = save_path / "cosp" / f"cosp_s{sub_id}_{state}_{fname}.npz"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=compute_cosp(data, fmin, fmax))

    print(f"sub-{sub_id}: done")


# ─── combine across subjects ──────────────────────────────────────────────────

def _load_npz(path: Path) -> np.ndarray:
    return np.load(path)["data"]


def combine_psd(save_path: Path, state: str, freq: str) -> None:
    out = save_path / "psd" / f"psd_{state}_{freq}.npz"
    if out.exists():
        return
    arrays = [_load_npz(f) for s in SUBJECT_IDS
              if (f := save_path / "psd" / f"psd_s{s}_{state}_{freq}.npz").exists()]
    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


def combine_cov(save_path: Path, state: str) -> None:
    out = save_path / "cov" / f"cov_{state}.npz"
    if out.exists():
        return
    arrays = [_load_npz(f) for s in SUBJECT_IDS
              if (f := save_path / "cov" / f"cov_s{s}_{state}.npz").exists()]
    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


def combine_cosp(save_path: Path, state: str, freq: str) -> None:
    out = save_path / "cosp" / f"cosp_{state}_{freq}.npz"
    if out.exists():
        return
    arrays = [_load_npz(f) for s in SUBJECT_IDS
              if (f := save_path / "cosp" / f"cosp_s{s}_{state}_{freq}.npz").exists()]
    if arrays:
        np.savez_compressed(out, data=np.array(arrays, dtype=object))


# ─── UMAP ─────────────────────────────────────────────────────────────────────

def _upper_tri(arr: np.ndarray) -> np.ndarray:
    """(n, p, p) -> (n, p*(p+1)/2)."""
    idx = np.triu_indices(arr.shape[-1])
    return arr[..., idx[0], idx[1]].reshape(len(arr), -1)


def build_umap_vectors(
    bids_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    psd_vecs, cov_vecs, cosp_vecs, labels = [], [], [], []

    for sub_id in SUBJECT_IDS:
        if not _vhdr(bids_path, sub_id).exists():
            continue
        try:
            state_data = load_epochs_umap(bids_path, sub_id)
        except Exception as e:
            print(f"  sub-{sub_id}: {e}")
            continue

        for state, data in state_data.items():
            psd_vecs.append(np.concatenate(
                [compute_psd(data, fmin, fmax) for fmin, fmax in FREQ_DICT.values()],
                axis=1))
            cov_vecs.append(_upper_tri(compute_cov(data)))
            cosp_vecs.append(np.concatenate(
                [_upper_tri(compute_cosp(data, fmin, fmax))
                 for fmin, fmax in FREQ_DICT.values()],
                axis=1))
            labels.extend([state] * len(data))

    psd_all  = np.concatenate(psd_vecs,  axis=0)
    cov_all  = np.concatenate(cov_vecs,  axis=0)
    cosp_all = np.concatenate(cosp_vecs, axis=0)

    return {
        "psd":  psd_all,
        "cov":  cov_all,
        "cosp": cosp_all,
        "all":  np.concatenate([psd_all, cov_all, cosp_all], axis=1),
    }, np.array(labels)


def plot_umaps(
    vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    titles = {"psd": "PSD (all bands)", "cov": "Covariance",
              "cosp": "Cospectrum (all bands)", "all": "PSD + Cov + Cosp"}

    for ax, (fname, title) in zip(axes.flatten(), titles.items()):
        print(f"  UMAP: {fname}")
        X = StandardScaler().fit_transform(vectors[fname])
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(X)

        for state in UMAP_STATES:
            mask = labels == state
            if mask.any():
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=UMAP_COLORS[state], s=3, alpha=0.4, rasterized=True)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(handles=[mpatches.Patch(color=UMAP_COLORS[s], label=s)
                            for s in UMAP_STATES], markerscale=3)

    fig.suptitle("UMAP — sleep stage separability by feature type", fontsize=15)
    plt.tight_layout()
    out = save_path / "umap_sleep_stages.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bids-path", type=Path, required=True,
                   help="Root of BIDS dataset (e.g. /home/alouis/scratch/dream_bids)")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Output directory for features (e.g. /home/alouis/scratch/dream_features)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for joblib (default: all CPUs)")
    return p.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    bids_path = args.bids_path
    save_path = args.save_path
    n_jobs    = args.n_jobs

    t0 = time()

    print("=== per-subject feature extraction ===")
    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(bids_path, save_path, sub_id)
        for sub_id in SUBJECT_IDS
    )

    print("=== combining subjects ===")
    Parallel(n_jobs=n_jobs)(
        delayed(combine_psd)(save_path, state, freq)
        for state, freq in product(STATE_LIST, FREQ_DICT)
    )
    Parallel(n_jobs=n_jobs)(
        delayed(combine_cov)(save_path, state) for state in STATE_LIST
    )
    Parallel(n_jobs=n_jobs)(
        delayed(combine_cosp)(save_path, state, freq)
        for state, freq in product(STATE_LIST, FREQ_DICT)
    )

    print("=== UMAP ===")
    vectors, labels = build_umap_vectors(bids_path)
    plot_umaps(vectors, labels, save_path)

    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")