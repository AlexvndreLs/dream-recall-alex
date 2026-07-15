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

# SFREQ (raw) conservé pour mat_eeg_to_bids.py qui lit les .mat originaux.
# Tous les scripts en aval (feat_extract, classify) utilisent SFREQ_PREPROC.

# Paramètres (utilisés dans preprocess_subject.py)
LINE_FREQ      = 50.0   # bruit de ligne secteur (France/Lyon)
HP_FREQ_FINAL  = 0.1    # HP final (matche hardware BIDS, préserve delta/SWS)
HP_FREQ_ICA    = 1.0    # HP temporaire pour le fit ICA uniquement (MNE trick)
SFREQ_TARGET   = 1000.0  # décimation finale (= SFREQ_PREPROC)
DECIMATE       = False   # si True : raw.resample(SFREQ_TARGET) dans preprocess_subject.py.
                          # False = réplication exacte thèse Arthur §1.2.3 (1000Hz, pas de downsampling).
                          # Remettre à True pour downsamplé 

SFREQ_PREPROC  = 1000.0                                # sfreq réelle en sortie de preprocess_subject.py,
                                                        # dépend de DECIMATE ci-dessus (1000.0 si False,
                                                        # SFREQ_TARGET si True) -> garder synchronisé à la main

EPOCH_DURATION = 30.0                                 # secondes (standard R&K / AASM)
N_SAMPLES      = int(SFREQ_PREPROC * EPOCH_DURATION)  # 30000 samples/epoch à 1000Hz (DECIMATE=False)

WINDOW         = 1000   # Welch : fenêtre Hanning 1000 samples = 1s à 1000Hz. Si DECIMATE=True
                        # un jour, il faudra ajuster WINDOW en conséquence (pas fait automatiquement).

OVERLAP        = 500
OVERLAP_COSP   = 0.75  # pyriemann CoSpectra rejette overlap=0.0 exactement (ValueError,
                        # cf cross_spectrum: `if not 0 < overlap < 1: raise ValueError`).
                        # 1e-6 -> step = int((1-1e-6)*1024) = 1023 sur 1024, soit 1 sample
                        # de chevauchement (0.1%) : négligeable, équivalent en pratique à
                        # l'absence d'overlap voulue par la thèse §1.2.6. Testé empiriquement
                        # (pyriemann==0.11).

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
    + [f"psd_osc_{b}" for b in FREQ_DICT]   # ratio puissance/aperiodic par bande (excès au-dessus du 1/f, FOOOF)
    + ["aperiodic"]                          # exposant aperiodic FOOOF (pente), par canal
    + ["cov"]                                # matrice de covariance temporelle (SCM)
    + [f"cosp_{b}"    for b in FREQ_DICT]   # cospectrum, par bande
    + ["perm_entropy", "higuchi_fd", "spec_entropy"]  # complexité (antropy), par canal
)

# Features matricielles : matrices SPD (n_epochs, 19, 19) classifiées en espace
# riemannien (TSclassifier). Le complément de FEATURE_KEYS est vectoriel et
# classifié par LDA euclidien. Dérivé de FREQ_DICT pour rester cohérent avec
# FEATURE_KEYS ; l'appartenance est déclarée ici plutôt que déduite du nom.
MATRIX_KEYS = ["cov"] + [f"cosp_{b}" for b in FREQ_DICT]

# Feature servant de référence pour compter les epochs par sujet/stade
# (n_trials_min global). Toutes les features d'un même sujet/stade ont le même
# nombre d'epochs ; 'cov' sert arbitrairement de compteur.
REF_KEY = "cov"

# Décalage appliqué aux graines des permutations pour qu'elles ne collisionnent
# pas avec celles des bootstraps (cf _seed dans classify.py).
PERM_SEED_OFFSET = 100_003
