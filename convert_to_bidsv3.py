import h5py
import numpy as np
import mne
import mne_bids
from pathlib import Path
import sys
from tqdm import tqdm


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

# Notation R&K (pas AASM : S3 et S4 restent séparés)
STAGE_MAP = {
    0: 'Sleep stage W',
    1: 'Sleep stage S1',
    2: 'Sleep stage S2',
    3: 'Sleep stage S3',   # Arthur fusionne S3+S4 en SWS au niveau des features (merge_S3_S4)
    4: 'Sleep stage S4',   # idem
    5: 'Sleep stage R',
   -1: 'UNKNOWN_m1',  # TODO: demander à Arthur
   -2: 'UNKNOWN_m2',  # TODO: demander à Arthur
}

# per_s19 est une copie de per_s29 — hypnogramme invalide
PER_BLACKLIST = {19}

# Sujets avec scoring jbe disponible
JBE_SUBJECTS = {1, 2, 4, 6, 10, 11, 14, 16, 18, 19, 23, 25, 26, 29, 32, 33, 35, 37}




def load_mat(path):
    with h5py.File(path, 'r') as f:
        data = f['m_data']
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        result = np.empty((n_channels, n_samples), dtype=np.float32)
        chunk_size = 1_000_000
        for i in tqdm(range(0, n_samples, chunk_size), desc=f"s{sub}", unit="chunk", ascii=True, ncols=80):
            result[:, i:i+chunk_size] = data[i:i+chunk_size, :].T.astype(np.float32)
        return result
    # Note float32 : RawArray préserve le dtype si les données sont déjà float32.
    # MNE peut upgrader en float64 en mémoire selon la version, mais write_raw_bids
    # écrit en fmt='single' (float32) par défaut le fichier FIF final est float32.


def load_hypno_annotations(path, prefix):
    with open(path) as f:
        stages = [int(l.strip()) for l in f if l.strip()]
    onsets = [float(i) for i in range(len(stages))]
    durations = [1.0] * len(stages)
    descriptions = [f"{prefix}/{STAGE_MAP.get(s, 'Sleep stage ?')}" for s in stages]
    return mne.Annotations(onsets, durations, descriptions)


sub = int(sys.argv[1])
sub_str = str(sub)

data = load_mat(DATA_PATH / 'data' / f's{sub_str}_sleep.mat')
info = mne.create_info(CH_NAMES, SFREQ, CH_TYPES)
info['line_freq'] = 50
raw = mne.io.RawArray(data, info, verbose=False)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Annotations : per sauf pour s19 (copie de s29), jbe si disponible
annotations = None

if sub not in PER_BLACKLIST:
    per_path = DATA_PATH / 'hypnograms' / f'hyp_per_s{sub_str}.txt'
    annotations = load_hypno_annotations(per_path, prefix='per')

if sub in JBE_SUBJECTS:
    jbe_path = DATA_PATH / 'hypnograms' / f'hyp_jbe_s{sub_str}.txt'
    jbe_annot = load_hypno_annotations(jbe_path, prefix='jbe')
    annotations = jbe_annot if annotations is None else annotations + jbe_annot

if annotations is not None:
    raw.set_annotations(annotations)

bids_path = mne_bids.BIDSPath(
    subject=sub_str.zfill(2), task='sleep',
    root=BIDS_PATH, datatype='eeg'
)
mne_bids.write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='Fif')
print("Done:", bids_path)
