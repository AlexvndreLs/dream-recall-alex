"""
Ce script sert à lire des données de sommeil brutes (au format MATLAB .mat),
à leur associer des classifications de stades de sommeil (hypnogrammes) et
à convertir le tout au format standard international BIDS

Ce travail est basé sur la bibliotheque MNE-BIDS, ainsi que
sur le travail precedent d'arthur : https://github.com/arthurdehgan/sleep
"""

import h5py
import numpy as np
import mne
import mne_bids
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from .config import CH_NAMES, CH_TYPES, SFREQ, STAGE_MAP, PER_BLACKLIST, JBE_SUBJECTS
# Pas d'infos sur les 3 derniers canaux M1,M2 et autre peut etre => message arthur
# arthur n'utilise que les 19 premiers

# Corrections ponctuelles d'alignement des hypnogrammes sur des epochs de 30s.
# (4, 'per') : hyp_per_s4 a perdu son 1er sample => offset {29} au lieu de {0}.
# Correction : préfixer d'1 sample (= hypno[0]) en tête.

HYPNO_FIXES = {
    (4, 'per'): 1,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--bids-path', type=Path, required=True)
    return parser.parse_args()


def load_mat(path, sub):
    with h5py.File(path, 'r') as f:
        data = f['m_data']
        n_samples = data.shape[0]
        result = np.empty((data.shape[1], n_samples), dtype=np.float32)  # attention inverse stockage entre mat et mne
        chunk_size = 1_000_000 
        # taille des blocs lus depuis le disque à la fois
        for i in tqdm(range(0, n_samples, chunk_size), desc=f"s{sub}", unit="chunk", ascii=True, ncols=80):
            result[:, i:i+chunk_size] = data[i:i+chunk_size, :].T.astype(np.float32)
            # on met dans le bon sens pour mne
            # float32 pareil pour pas exploser le cluster
        return result

    # note float32 : RawArray préserve le dtype si les données sont déjà float32.
    # MNE upgrade quand meme en float64 mais write_raw_bids
    # écrit en fmt='single' (float32) par défaut le fichier Brainvision final est float32
    # on limite un peu la ram lors du chargement


def load_hypno_annotations(path, prefix, sub):
    with open(path) as f:
        stages = [int(line.strip()) for line in f if line.strip()]
    # retire les espaces/sauts de ligne + ignore les lignes vides +
    # convertit le chiffre trouvé en entier (int) + stocke le tout dans une liste

    # Correction alignement epoch 30s : préfixe les samples manquants en tête si nécessaire
    n_prepend = HYPNO_FIXES.get((sub, prefix), 0)
    if n_prepend:
        stages = [stages[0]] * n_prepend + stages

    onsets = [float(i) for i in range(len(stages))]
    # liste des temps de début en secondes pour chaque stade
    durations = [1.0] * len(stages)  # chaque annotation dure 1 seconde (a la base je crois 30s mais apres x30)
    descriptions = [f"{prefix}/{STAGE_MAP.get(s, 'Sleep stage ?')}" for s in stages]
    return mne.Annotations(onsets, durations, descriptions)


def main():
    args = parse_args()
    sub = args.subject
    sub_str = str(sub)
    data_path = args.data_path
    bids_path = args.bids_path

    data = load_mat(data_path / 'data' / f's{sub_str}_sleep.mat', sub)
    data *= 1e-6  # µV => V pour mne qui attend des V
    info = mne.create_info(CH_NAMES, SFREQ, CH_TYPES)
    info['line_freq'] = 50  # 50 Hz standard européen
    raw = mne.io.RawArray(data, info, verbose=False)

    # arthur n'utilise pas les coordonées dans son code mais systeme 10-20 via code matlab donc pareil mne
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

    # Annotations : per sauf pour s19 (copie de s29), jbe si disponible
    annotations = None

    if sub not in PER_BLACKLIST:
        per_path = data_path / 'hypnograms' / f'hyp_per_s{sub_str}.txt'
        annotations = load_hypno_annotations(per_path, prefix='per', sub=sub)

    if sub in JBE_SUBJECTS:
        jbe_path = data_path / 'hypnograms' / f'hyp_jbe_s{sub_str}.txt'
        jbe_annot = load_hypno_annotations(jbe_path, prefix='jbe', sub=sub)
        annotations = jbe_annot if annotations is None else annotations + jbe_annot
        # ca reviens a si juste l'un on le prends sinon les 2

    if annotations is not None:
        raw.set_annotations(annotations)

    bids_output = mne_bids.BIDSPath(
        subject=sub_str.zfill(2), task='sleep',  # guidelines BIDS
        root=bids_path, datatype='eeg'
    )

    mne_bids.write_raw_bids(raw, bids_output, overwrite=True, allow_preload=True, format='BrainVision')

    # Patch sidecar JSON avec métadonnées issues du papier (write_raw_bids ne supporte pas extra_infos)
    # Source : Dehgan et al., BrainAmp (Brain Products), électrodes Ag/AgCl, système 10-20 étendu
    # Référence : tip of the nose ; ground : forehead ; filtre HP : 0.1 Hz ; impédance < 5 kΩ
    sidecar_path = bids_output.fpath.with_suffix('.json')
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    sidecar.update({
        'Manufacturer': 'Brain Products',
        'ManufacturersModelName': 'BrainAmp',
        'EEGReference': 'tip of the nose',
        'EEGGround': 'forehead',
        'EEGPlacementScheme': 'extended 10-20',
        'HardwareFilters': {
            'Highpass RC filter': {'Half amplitude cutoff (Hz)': 0.1},
            'Lowpass anti-aliasing filter': 'present'  # le papier ne donne pas la fréquence exacte du passe-bas, juste sa présence
        },
        'SoftwareFilters': 'n/a',
        'RecordingType': 'continuous',
        'AmplifierGain': 12500
    })  # issu du papier d'arthur
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=4)  # pour que ca reste lisible

    print("Done:", bids_output)


if __name__ == "__main__":
    main()
