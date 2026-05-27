import h5py
import numpy as np
from pathlib import Path

data_dir = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/data")
hypno_dir = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/hypnograms")

print(f"{'Sujet':<10} | {'EEG (s)':<12} | {'Hypno (s)':<12} | {'Différence':<12}")
print("-" * 55)

for raw_file in sorted(data_dir.glob("*_sleep.mat")):
    subject_id = raw_file.name.split('_')[0]
    
    # Trouver l'hypnogramme correspondant (hyp_per_ ou hyp_jbe_)
    hypno_path = None
    for prefix in ["hyp_per_", "hyp_jbe_"]:
        p = hypno_dir / f"{prefix}{subject_id}.txt"
        if p.exists():
            hypno_path = p
            break
    
    if hypno_path:
        with h5py.File(raw_file, 'r') as f:
            eeg_dur = f['m_data'].shape[0] / 1000.0
        
        with open(hypno_path, 'r') as f:
            hypno_dur = len(f.readlines())
            
        diff = hypno_dur - eeg_dur
        # Affichage systématique pour tous les sujets
        print(f"{subject_id:<10} | {eeg_dur:>12.2f} | {hypno_dur:>12} | {diff:>12.2f}")
    else:
        print(f"{subject_id:<10} | Hypnogramme introuvable")
