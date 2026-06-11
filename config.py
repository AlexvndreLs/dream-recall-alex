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
 
# ─── sleep stages: single atomic segmentation, derived groupings ──────────────
 
# Raw scorer labels -> atomic codes. Each 30s epoch belongs to exactly ONE
# atomic stage. Features are computed once per atomic stage; classification
# states and UMAP states are both obtained by concatenating atomic arrays
# (no recomputation, no double I/O).
STAGE_LABEL_TO_ATOMIC = {
    "Sleep stage S1": "S1",
    "Sleep stage S2": "S2",
    "Sleep stage S3": "S3",
    "Sleep stage S4": "S4",
    "Sleep stage R":  "REM",
}
ATOMIC_STAGES = ["S1", "S2", "S3", "S4", "REM"]
 
# Classification states (SWS = S3+S4, NREM = S1+S2+S3+S4 — overlaps S2/SWS)
CLASSIFICATION_GROUPS = {
    "S2":   ["S2"],
    "SWS":  ["S3", "S4"],
    "REM":  ["REM"],
    "NREM": ["S1", "S2", "S3", "S4"],
}
STATE_LIST = list(CLASSIFICATION_GROUPS)
 
# UMAP states (atomic, no overlap)
UMAP_GROUPS = {
    "S1":  ["S1"],
    "S2":  ["S2"],
    "SWS": ["S3", "S4"],
    "REM": ["REM"],
}
UMAP_STATES = list(UMAP_GROUPS)
UMAP_COLORS = {"S1": "#E69F00", "S2": "#56B4E9", "SWS": "#009E73", "REM": "#CC79A7"}
 
# ─── feature set ────────────────────────────────────────────────────────────
 
FOOOF_FREQ_RANGE = (1, 45)  # full-spectrum range for aperiodic fitting
 
FEATURE_KEYS = (
    [f"psd_{b}" for b in FREQ_DICT]      # raw band power
    + [f"psd_osc_{b}" for b in FREQ_DICT]  # 1/f-corrected (oscillatory) band power
    + ["aperiodic"]                       # FOOOF exponent (slope), per channel
    + ["cov"]                             # time covariance
    + [f"cosp_{b}" for b in FREQ_DICT]    # cospectrum, per band
)
 
SUBJECT_IDS = [f"{i:02d}" for i in range(1, 39)]