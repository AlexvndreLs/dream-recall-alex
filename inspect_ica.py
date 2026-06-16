"""Script local d'inspection visuelle des composantes ICA.

À lancer en local (pas sur le cluster) après le preprocessing,
pour valider que les composantes rejetées par find_bads_eog /
find_bads_muscle sont bien des artefacts, et calibrer le seuil
threshold de find_bads_muscle si nécessaire.

Prérequis :
    - preprocess_subject_v2.py doit avoir tourné pour le sujet cible
    - L'objet ICA est dans derivatives/ica/sub-XX_task-sleep_ica.fif
    - Le BIDS brut est dans dream_bids/ (25 canaux, 1000Hz)

Usage :
    python inspect_ica.py 5 \\
        --bids-path  /home/alouis/scratch/dream_bids \\
        --deriv-root /home/alouis/scratch/dream_bids/derivatives

Ce que fait ce script :
    1. Charge le raw BIDS brut (25 canaux, 1000Hz)
    2. Recharge l'objet ICA sauvegardé
    3. Affiche les composantes rejetées (topographies, time series, spectre)
    4. Affiche toutes les composantes pour comparaison
    5. Permet de modifier manuellement ica.exclude si besoin

Les figures s'affichent de manière interactive (matplotlib backend GUI).
"""

import argparse
from pathlib import Path

import mne
import mne_bids

from config_v2 import CH_NAMES, HP_FREQ_ICA, N_EEG


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=int)
    p.add_argument('--bids-path', type=Path, required=True,
                   help="Racine du BIDS brut (dream_bids/)")
    p.add_argument('--deriv-root', type=Path, required=True,
                   help="Racine derivatives (contient ica/)")
    p.add_argument('--hp-freq', type=float, default=HP_FREQ_ICA,
                   help=f"HP pour la visualisation ICA (défaut: {HP_FREQ_ICA}Hz)")
    return p.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args    = parse_args()
    sub_str = str(args.subject).zfill(2)

    # 1. Charge le raw BIDS brut
    bids_input = mne_bids.BIDSPath(
        subject=sub_str, task='sleep',
        root=args.bids_path, datatype='eeg',
    )
    raw = mne_bids.read_raw_bids(bids_input, verbose=False)
    raw.load_data()

    # HP 1Hz pour visualisation cohérente avec le fit ICA
    raw_vis = raw.copy().filter(l_freq=args.hp_freq, h_freq=None, verbose=False)

    # 2. Recharge l'objet ICA
    ica_path = args.deriv_root / "ica" / f"sub-{sub_str}_task-sleep_ica.fif"
    if not ica_path.exists():
        raise FileNotFoundError(
            f"ICA non trouvé : {ica_path}\n"
            "Lancer preprocess_subject_v2.py d'abord."
        )
    ica = mne.preprocessing.read_ica(ica_path)

    print(f"sub-{sub_str} — {ica.n_components_} composantes ICA")
    print(f"Composantes exclues automatiquement : {ica.exclude}")

    # 3. Topographies + time series des composantes exclues
    if ica.exclude:
        print("\n--- Composantes rejetées (topographies) ---")
        ica.plot_components(picks=ica.exclude, title=f"sub-{sub_str} — rejetées")

        print("\n--- Propriétés détaillées des composantes rejetées ---")
        ica.plot_properties(raw_vis, picks=ica.exclude)
    else:
        print("Aucune composante rejetée automatiquement.")

    # 4. Toutes les composantes pour comparaison
    print("\n--- Toutes les composantes (sources temporelles) ---")
    ica.plot_sources(raw_vis, title=f"sub-{sub_str} — toutes composantes")

    print("\n--- Toutes les topographies ---")
    ica.plot_components(title=f"sub-{sub_str} — toutes topographies")

    # 5. Signal avant/après ICA sur les 19 EEG
    raw_eeg = raw_vis.copy().pick(CH_NAMES[:N_EEG])
    raw_clean = ica.apply(raw_eeg.copy(), verbose=False)

    print("\n--- Signal EEG avant ICA ---")
    raw_eeg.plot(title=f"sub-{sub_str} — avant ICA", scalings='auto')

    print("\n--- Signal EEG après ICA ---")
    raw_clean.plot(title=f"sub-{sub_str} — après ICA", scalings='auto')

    print("\nInspection terminée.")
    print("Pour modifier les composantes rejetées, éditer ica.exclude manuellement")
    print("et relancer preprocess_subject_v2.py avec les nouveaux paramètres.")
    mne.viz.plt.show()
