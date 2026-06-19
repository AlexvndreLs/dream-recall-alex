CH_NAMES = [
    'Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4',
    'Fp1', 'Fp2', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4',
    'FC1', 'FC2', 'CP1', 'CP2',
    'EOG_L', 'EOG_R', 'EMG_chin',
    'misc1', 'misc2', 'misc3'
]

CH_TYPES = ['eeg'] * 19 + ['eog', 'eog', 'emg'] + ['misc'] * 3
# canaux 23-25 (misc1/misc2/misc3) : probablement M1/M2 + un autre,
# identité exacte non confirmée -> message Arthur en attente
# Arthur n'utilise que les 19 premiers (EEG)

SFREQ = 1000.0  # fréquence d'échantillonnage raw (BrainAmp, hardware)

# Notation R&K (pas AASM : S3 et S4 restent séparés)
# merge_S3_S4 dans le repo d'Arthur est un post-processing sur features,
# pas sur le signal brut
STAGE_MAP = {
    0: 'Sleep stage W',
    1: 'Sleep stage S1',
    2: 'Sleep stage S2',
    3: 'Sleep stage S3',   # Arthur fusionne S3+S4 en SWS au niveau features
    4: 'Sleep stage S4',   # idem
    5: 'Sleep stage R',
   -1: 'UNKNOWN_m1',       # ignoré dans load_hypno de utils.py d'Arthur
   -2: 'UNKNOWN_m2',       # ignoré dans load_hypno de utils.py d'Arthur
}

# ─── labels HR/LR ─────────────────────────────────────────────────────────────
#
# Source : tableau Arthur (Riemannian_Dream_Recall_Subject_numbers.xlsx)
# + réponse directe Arthur (message 16/06/2026) :
#   "Les sujets 1 à 18 sont HR, les sujets 19, 20 et 23-38 sont LR.
#    Les sujets 21 et 22 sont exclus (raison inconnue, investigation en cours)."
#
# Encodage cohérent avec INIT_LABELS = [0]*18 + [1]*18 dans le repo Arthur :
#   SUBJECT_LIST = [sujets LR en premier, puis sujets HR]
#   -> position 0-17 = LR (label 0), position 18-35 = HR (label 1)
# Mais ici on encode explicitement par numéro de sujet pour plus de clarté.
#
# Note s19 : hyp_per_s19.txt est une copie de hyp_per_s29.txt sur le disque
# (bug détecté à l'audit, confirmé byte-for-byte). s19 est un LR valide avec
# des données EEG correctes ; seul l'hypnogramme per est corrompu.
# Solution : utiliser l'hypnogramme jbe de s19 (disponible dans JBE_SUBJECTS).

HR_SUBJECTS       = set(range(1, 19))               # sujets 1-18  (High Recallers)
LR_SUBJECTS       = {19, 20} | set(range(23, 39))   # sujets 19,20,23-38 (Low Recallers)
EXCLUDED_SUBJECTS = {21, 22}                         # exclus de l'analyse, raison inconnue

# Liste ordonnée des 36 sujets inclus dans l'analyse (LR d'abord, puis HR),
# ordre cohérent avec INIT_LABELS d'Arthur (positions 0-17 = LR, 18-35 = HR)
SUBJECT_LIST_ORDERED = (
    sorted(LR_SUBJECTS) +  # [19, 20, 23, 24, ..., 38]  -> labels 0
    sorted(HR_SUBJECTS)    # [1, 2, 3, ..., 18]          -> labels 1
)

# Labels binaires dans le même ordre que SUBJECT_LIST_ORDERED
# 0 = LR, 1 = HR  (cohérent avec INIT_LABELS = [0]*18 + [1]*18 d'Arthur)
SUBJECT_LABELS = [0] * len(LR_SUBJECTS) + [1] * len(HR_SUBJECTS)

# IDs BIDS de tous les sujets disponibles (38 fichiers .mat)
# sujets 21 et 22 inclus ici car les fichiers existent sur disque,
# mais exclus des analyses via EXCLUDED_SUBJECTS
SUBJECT_IDS = [f"{i:02d}" for i in range(1, 39)]

# IDs BIDS des sujets inclus dans l'analyse uniquement (36 sujets)
SUBJECT_IDS_ANALYSIS = [
    f"{i:02d}" for i in sorted(HR_SUBJECTS | LR_SUBJECTS)
]

# per_s19 est une copie de per_s29 sur le disque -> blacklisté pour scorer per
# s19 a un hypnogramme jbe valide (cf JBE_SUBJECTS)
PER_BLACKLIST     = {19}
PER_BLACKLIST_STR = {f"{i:02d}" for i in PER_BLACKLIST}

# Sujets avec scoring jbe disponible
JBE_SUBJECTS     = {1, 2, 4, 6, 10, 11, 14, 16, 18, 19, 23, 25, 26, 29, 32, 33, 35, 37}
JBE_SUBJECTS_STR = {f"{i:02d}" for i in JBE_SUBJECTS}


# ─── preprocessing params ─────────────────────────────────────────────────────
#
# Mis à jour après preprocessing (preprocess_subject_v2.py) :
# les données passent de 1000Hz à 250Hz -> N_SAMPLES et WINDOW mis à jour.
# SFREQ (raw) conservé pour mat_eeg_to_bids_v2.py qui lit les .mat originaux.
# Tous les scripts en aval (feat_extract, classify) utilisent SFREQ_PREPROC.

# Paramètres ZapLine / filtrage / ICA (utilisés dans preprocess_subject_v2.py)
LINE_FREQ      = 50.0   # bruit de ligne secteur (France/Lyon)
HP_FREQ_FINAL  = 0.1    # HP final (matche hardware BIDS, préserve delta/SWS)
HP_FREQ_ICA    = 1.0    # HP temporaire pour le fit ICA uniquement (MNE trick)
SFREQ_TARGET   = 250.0  # décimation finale (= SFREQ_PREPROC)

SFREQ_PREPROC  = 250.0                                # après décimation dans preprocess_subject.py
EPOCH_DURATION = 30.0                                 # secondes (standard R&K / AASM)
N_SAMPLES      = int(SFREQ_PREPROC * EPOCH_DURATION)  # 7500 samples par epoch à 250Hz
WINDOW         = 250   # Welch : fenêtre 1s Hanning, no overlap — thesis §1.2.5
                        # (1s à 250Hz = 250 samples ; même durée temporelle qu'à 1000Hz)
                        # (au lieu de 1000 à 1000Hz, même durée en secondes)
OVERLAP        = 0


# ─── feature extraction params ────────────────────────────────────────────────

N_EEG = 19  # les 19 premiers canaux de CH_NAMES sont l'EEG

FREQ_DICT = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (11, 16),
    "beta":  (17, 35),
}

FOOOF_FREQ_RANGE = (1, 45)  # plage complète pour le fit aperiodic FOOOF
                             # 45Hz < Nyquist (125Hz à 250Hz) -> pas d'aliasing


# ─── sleep stages : segmentation atomique + groupements dérivés ───────────────

# Labels scorer -> codes atomiques. Chaque epoch 30s appartient à exactement
# UN stade atomique. Les features sont calculées une fois par stade atomique ;
# les états de classification et UMAP sont obtenus par concaténation des
# tableaux atomiques (pas de recalcul, pas de double I/O).
STAGE_LABEL_TO_ATOMIC = {
    "Sleep stage S1": "S1",
    "Sleep stage S2": "S2",
    "Sleep stage S3": "S3",
    "Sleep stage S4": "S4",
    "Sleep stage R":  "REM",
}
ATOMIC_STAGES = ["S1", "S2", "S3", "S4", "REM"]

# États de classification (SWS = S3+S4, NREM = S1+S2+S3+S4)
CLASSIFICATION_GROUPS = {
    "S2":   ["S2"],
    "SWS":  ["S3", "S4"],
    "REM":  ["REM"],
    "NREM": ["S1", "S2", "S3", "S4"],
}
STATE_LIST = list(CLASSIFICATION_GROUPS)

# États UMAP (atomiques, sans overlap)
UMAP_GROUPS = {
    "S1":  ["S1"],
    "S2":  ["S2"],
    "SWS": ["S3", "S4"],
    "REM": ["REM"],
}
UMAP_STATES = list(UMAP_GROUPS)

# Palette Okabe-Ito (colorblind-safe, standard Nature/Wong 2011)
UMAP_COLORS = {"S1": "#E69F00", "S2": "#56B4E9", "SWS": "#009E73", "REM": "#CC79A7"}


# ─── feature set ──────────────────────────────────────────────────────────────

FEATURE_KEYS = (
    [f"psd_{b}"       for b in FREQ_DICT]   # puissance spectrale brute par bande
    + [f"psd_osc_{b}" for b in FREQ_DICT]   # puissance oscillatoire corrigée 1/f (FOOOF)
    + ["aperiodic"]                          # exposant aperiodic FOOOF (pente), par canal
    + ["cov"]                                # matrice de covariance temporelle (SCM)
    + [f"cosp_{b}"    for b in FREQ_DICT]   # cospectrum, par bande
)
