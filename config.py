CH_NAMES = [
    'Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4',
    'Fp1', 'Fp2', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4',
    'FC1', 'FC2', 'CP1', 'CP2',
    'EOG_L', 'EOG_R', 'EMG_chin',
    'misc1', 'misc2', 'misc3'
]

CH_TYPES = ['eeg'] * 19 + ['eog', 'eog', 'emg'] + ['misc'] * 3

SFREQ = 1000.0

STAGE_MAP = {
    0: 'Sleep stage W',
    1: 'Sleep stage S1',
    2: 'Sleep stage S2',
    3: 'Sleep stage S3',
    4: 'Sleep stage S4',
    5: 'Sleep stage R',
   -1: 'UNKNOWN_m1',
   -2: 'UNKNOWN_m2',
}

PER_BLACKLIST = {19}

JBE_SUBJECTS = {1, 2, 4, 6, 10, 11, 14, 16, 18, 19, 23, 25, 26, 29, 32, 33, 35, 37}