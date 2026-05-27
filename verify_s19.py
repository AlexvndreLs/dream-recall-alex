import h5py
import numpy as np
from pathlib import Path

# Chemins d'accès (adaptés selon vos structures dans SESSION_SUMMARY_2026-05-26.md)
raw_path = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/data/s19_sleep.mat")
hypno_path = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/hypnograms/hyp_per_s19.txt")

# 1. Calcul de la durée EEG (en secondes)
with h5py.File(raw_path, 'r') as f:
    n_samples = f['m_data'].shape[0]  # n_samples, n_channels[cite: 2]
    eeg_duration_s = n_samples / 1000.0
    print(f"Durée EEG s19 : {eeg_duration_s:.2f} secondes")

# 2. Calcul de la durée hypnogramme (1 ligne = 1 seconde)[cite: 2]
with open(hypno_path, 'r') as f:
    hypno_duration_s = len(f.readlines())
    print(f"Durée Hypnogramme s19 : {hypno_duration_s} secondes")

diff = hypno_duration_s - eeg_duration_s
print(f"Différence : {diff:.2f} secondes")
