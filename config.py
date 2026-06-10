CH_NAMES = [
    'Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4',
    'Fp1', 'Fp2', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4',
    'FC1', 'FC2', 'CP1', 'CP2',
    'EOG_L', 'EOG_R', 'EMG_chin',
    'misc1', 'misc2', 'misc3'
]

CH_TYPES = ['eeg'] * 19 + ['eog', 'eog', 'emg'] + ['misc'] * 3 
#Pas d'infos sur les 3 derniers canaux M1,M2 et autre peut etre => message arthur
#arthur n'utilise que les 19 premiers

SFREQ = 1000.0

# Notation R&K (pas AASM : S3 et S4 restent séparés)
# merge_S3_S4 dans le repo d'Arthur est un post-processing sur features, pas sur le signal brut
STAGE_MAP = {
    0: 'Sleep stage W',
    1: 'Sleep stage S1',
    2: 'Sleep stage S2',
    3: 'Sleep stage S3',   # Arthur fusionne S3+S4 en SWS au niveau des features (merge_S3_S4)
    4: 'Sleep stage S4',   # idem
    5: 'Sleep stage R',
   -1: 'UNKNOWN_m1',  # Todo: demander à Arthur (ignoré dans load_hypno de utils.py)
   -2: 'UNKNOWN_m2',  # Todo: demander à Arthur (ignoré dans load_hypno de utils.py)
}

# per_s19 est une copie de per_s29 !!!
PER_BLACKLIST = {19}

# Sujets avec scoring jbe disponible
JBE_SUBJECTS = {1, 2, 4, 6, 10, 11, 14, 16, 18, 19, 23, 25, 26, 29, 32, 33, 35, 37}

 
# ─── feature extraction params (extract_features.py) ──────────────────────────
 
EPOCH_DURATION = 30.0  # seconds
N_SAMPLES = int(SFREQ * EPOCH_DURATION)  # 30 000
 
WINDOW = 1000   # Welch: 1s Hanning, no overlap — thesis §1.2.5
OVERLAP = 0
 
N_EEG = 19  # first 19 channels are EEG
 
FREQ_DICT = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (11, 16),
    "beta":  (17, 35),
}
 
# Classification states (SWS/NREM aggregate epochs from multiple raw stages)
STAGE_TO_STATES = {
    "Sleep stage S1": ["NREM"],
    "Sleep stage S2": ["S2", "NREM"],
    "Sleep stage S3": ["SWS", "NREM"],
    "Sleep stage S4": ["SWS", "NREM"],
    "Sleep stage R":  ["REM"],
}
STATE_LIST = ["S2", "SWS", "REM", "NREM"]
 
# Atomic states for UMAP (no overlap)
STAGE_TO_UMAP = {
    "Sleep stage S1": "S1",
    "Sleep stage S2": "S2",
    "Sleep stage S3": "SWS",
    "Sleep stage S4": "SWS",
    "Sleep stage R":  "REM",
}
UMAP_STATES = ["S1", "S2", "SWS", "REM"]
UMAP_COLORS = {
    "S1":  "#E69F00",  # orange
    "S2":  "#56B4E9",  # sky blue
    "SWS": "#009E73",  # bluish green
    "REM": "#CC79A7",  # reddish purple
}

SUBJECT_IDS = [f"{i:02d}" for i in range(1, 39)]
 