"""Test isolé : window="hamming" (code réel d'Arthur, utils.py::computePSD)
vs window="hann" (notre choix actuel, basé sur le texte de la thèse), pour la
feature psd_sigma en S2 uniquement.

Ne touche AUCUN fichier existant. Recalcule juste la PSD sigma/S2 par sujet
avec les DEUX fenêtres, produit un .npz de features (n_subjects, n_epochs, 19)
pour chacune, puis on pourra les passer dans classify.py pour comparer les
p-values. But : savoir si l'ecart de p-value avec Arthur vient de la fenetre
(hann/hamming) ou d'autre chose (comptage d'epochs, etc.).

Reproduit exactement la logique de feat_extract_umap_fooof_v4.py
(load_epochs_by_atomic_stage + compute_psd_spectrum + band_power) mais en
faisant varier uniquement le parametre window.

Usage:
    python test_hamming_vs_hann_sigma_s2.py \\
        --deriv-path /home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica \\
        --out-dir /home/alouis/scratch/test_window_sigma_s2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from config_v3 import (
    CH_NAMES, N_EEG, SFREQ_PREPROC, WINDOW, OVERLAP, N_SAMPLES,
    FREQ_DICT, FOOOF_FREQ_RANGE, STAGE_LABEL_TO_ATOMIC,
    SUBJECT_IDS_ANALYSIS, PER_BLACKLIST_STR, JBE_SUBJECTS_STR,
)

SF = int(SFREQ_PREPROC)
STATE = "S2"
FMIN, FMAX = FREQ_DICT["sigma"]  # (11, 16)


def _vhdr(deriv_path, sub_id):
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_eeg.vhdr")


def _events(deriv_path, sub_id):
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_events.tsv")


def _choose_scorer(sub_id):
    if sub_id not in PER_BLACKLIST_STR:
        return "per"
    if sub_id in JBE_SUBJECTS_STR:
        return "jbe"
    raise ValueError(f"sub-{sub_id}: no valid scorer")


def load_s2_epochs(deriv_path, sub_id):
    """Reproduit load_epochs_by_atomic_stage mais pour S2 seulement."""
    raw = mne.io.read_raw_brainvision(_vhdr(deriv_path, sub_id),
                                       preload=True, verbose=False)
    raw.pick(CH_NAMES[:N_EEG])
    assert raw.info["sfreq"] == SFREQ_PREPROC, (
        f"sub-{sub_id}: sfreq {raw.info['sfreq']} != {SFREQ_PREPROC}")
    n_total = raw.n_times

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"
    df = pd.read_csv(_events(deriv_path, sub_id), sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = (df[df["stage"].isin(STAGE_LABEL_TO_ATOMIC)]
          .sort_values("sample").reset_index(drop=True))

    out = []
    i = 0
    while i + 29 < len(df):
        block = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages = block["stage"].values
        if not (np.all(samples == samples[0] + np.arange(30) * SF) and
                np.all(stages == stages[0])):
            i += 1
            continue
        if STAGE_LABEL_TO_ATOMIC[stages[0]] != STATE:
            i += 30
            continue
        end = int(samples[0]) + N_SAMPLES
        if end > n_total:
            raise ValueError(f"sub-{sub_id}: epoch depasse fin fichier")
        out.append(raw.get_data(start=int(samples[0]), stop=end))
        i += 30
    if not out:
        return None
    return np.stack(out)  # (n_epochs, 19, 30000)


def psd_sigma(data, window):
    """(n_epochs, 19, 30000) -> (n_epochs, 19) puissance sigma moyenne."""
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=SF, fmin=FOOOF_FREQ_RANGE[0], fmax=FOOOF_FREQ_RANGE[1],
        n_fft=WINDOW, n_overlap=OVERLAP, n_per_seg=WINDOW,
        window=window, verbose=False,
    )
    mask = (freqs >= FMIN) & (freqs <= FMAX)
    return psds[..., mask].mean(axis=-1)  # (n_epochs, 19)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_sub_hann = []
    per_sub_hamming = []
    kept_subjects = []

    for sub_id in SUBJECT_IDS_ANALYSIS:
        epochs = load_s2_epochs(args.deriv_path, sub_id)
        if epochs is None:
            print(f"sub-{sub_id}: aucune epoch S2, skip")
            continue
        feat_hann = psd_sigma(epochs, "hann")
        feat_hamming = psd_sigma(epochs, "hamming")
        per_sub_hann.append(feat_hann)
        per_sub_hamming.append(feat_hamming)
        kept_subjects.append(sub_id)

        # correlation intra-sujet entre les deux fenetres, pour info
        r = np.corrcoef(feat_hann.ravel(), feat_hamming.ravel())[0, 1]
        print(f"sub-{sub_id}: {epochs.shape[0]:4d} epochs S2 | "
              f"corr hann/hamming = {r:.4f} | "
              f"ratio moyen hamming/hann = {feat_hamming.mean()/feat_hann.mean():.4f}")

    # Sauvegarde au format attendu par classify.py :
    # un objet array de longueur n_subjects, chaque element (n_epochs, 19).
    for name, per_sub in [("hann", per_sub_hann), ("hamming", per_sub_hamming)]:
        arr = np.empty(len(per_sub), dtype=object)
        for k, v in enumerate(per_sub):
            arr[k] = v
        out = args.out_dir / f"psd_sigma_{STATE}_{name}.npz"
        np.savez_compressed(out, data=arr, subjects=np.array(kept_subjects))
        print(f"-> {out}  ({len(per_sub)} sujets)")

    print()
    print("Note: ces .npz ont un format 'brut' (features par sujet), PAS le format")
    print("de sortie de classify.py. Pour comparer les p-values, il faut les passer")
    print("dans le meme pipeline de classification. Dis-moi si tu veux ce wrapper.")
