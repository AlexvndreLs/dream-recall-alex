import h5py
import numpy as np
import mne
import mne_bids
from pathlib import Path

DATA_PATH = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data")
BIDS_PATH = Path("/home/alouis/scratch/dream_bids")

CH_NAMES = [
    'Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4',
    'Fp1', 'Fp2', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4',
    'FC1', 'FC2', 'CP1', 'CP2',
    'EOG_L', 'EOG_R', 'EMG_chin',
    'misc1', 'misc2', 'misc3'
]
CH_TYPES = ['eeg'] * 19 + ['eog', 'eog', 'emg'] + ['stim'] * 3
SFREQ = 1000.0

STAGE_MAP = {
    0: 'Sleep stage W',
    1: 'Sleep stage N1',
    2: 'Sleep stage N2',
    3: 'Sleep stage N3',
    4: 'Sleep stage N3',
    5: 'Sleep stage R',
   -1: 'Sleep stage ?',
   -2: 'BAD_movement',
}

COORDS = np.array([
    [0.714, 0, 0.7],
    [6.12e-17, 0, 1],
    [-0.714, -8.74e-17, 0.7],
    [4.55e-17, 0.744, 0.668],
    [4.55e-17, -0.744, 0.668],
    [6.09e-17, 0.995, -0.103],
    [6.09e-17, -0.995, -0.103],
    [0.95, 0.309, -0.0471],
    [0.95, -0.309, -0.0471],
    [-0.95, 0.309, -0.0471],
    [-0.95, -0.309, -0.0471],
    [0.677, 0.568, 0.468],
    [0.676, -0.567, 0.471],
    [-0.677, 0.568, 0.468],
    [-0.676, -0.567, 0.471],
    [0.381, 0.381, 0.843],
    [0.381, -0.381, 0.843],
    [-0.381, 0.381, 0.843],
    [-0.381, -0.381, 0.843],
])

def load_mat(path):
    with h5py.File(path, 'r') as f:
        data = f['m_data']
        # lecture par chunks de 1M samples pour limiter le pic mémoire
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        result = np.empty((n_channels, n_samples), dtype=np.float32)
        chunk_size = 1_000_000
        for i in range(0, n_samples, chunk_size):
            result[:, i:i+chunk_size] = data[i:i+chunk_size, :].T.astype(np.float32)
        return result

def load_hypno(path):
    with open(path) as f:
        return [int(l.strip()) for l in f if l.strip()]

def hypno_to_annotations(hypno):
    onsets, durations, descriptions = [], [], []
    for i, stage in enumerate(hypno):
        onsets.append(float(i))
        durations.append(1.0)
        descriptions.append(STAGE_MAP.get(stage, 'Sleep stage ?'))
    return mne.Annotations(onsets, durations, descriptions)

import sys
sub = sys.argv[1]
data = load_mat(DATA_PATH / 'data' / f's{sub}_sleep.mat')
hypno = load_hypno(DATA_PATH / 'hypnograms' / f'hyp_per_s{sub}.txt')

info = mne.create_info(CH_NAMES, SFREQ, CH_TYPES)
info['line_freq'] = 50
raw = mne.io.RawArray(data, info, verbose=False)
raw._data = np.array(raw._data, dtype=np.float32)

ch_pos_m = {CH_NAMES[i]: COORDS[i] * 0.095 for i in range(19)} #0.095 ref mne tuto / valeur par defaut
montage = mne.channels.make_dig_montage(ch_pos=ch_pos_m, coord_frame='head')
raw.set_montage(montage)
raw.set_annotations(hypno_to_annotations(hypno))

bids_path = mne_bids.BIDSPath(subject=sub.zfill(2), task='sleep',
                               root=BIDS_PATH, datatype='eeg')
mne_bids.write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='Fif')
print("Done:", bids_path)
