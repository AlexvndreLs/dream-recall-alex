"""
Preprocessing offline des données EEG sommeil (dataset Ruby/Eichenlaub, CRNL Lyon)
avant extraction de features (feat_extract_umap_fooof.py) et entraînement réseau DL.

Ce script produit DEUX sorties BIDS derivatives par sujet :

  - derivatives/preprocessed-ica/   : pipeline complet AVEC ICA
                                       -> destiné à feat_extract (PSD/cov/cosp)
  - derivatives/preprocessed-noica/ : pipeline identique SANS l'étape ICA
                                       -> destiné au réseau DL, pour permettre
                                          une ablation empirique ICA vs no-ICA
                                          sur la tâche HR/LR (cf discussion :
                                          l'argument "ICA détruit le signal
                                          EOG = signature REM" est valide en
                                          général, mais sa pertinence pour
                                          dream recall spécifiquement n'est
                                          pas établie -> on laisse les deux
                                          branches trancher empiriquement)

Ordre des opérations (important, cf justifications inline) :
  ZapLine 50Hz -> HP 0.1Hz (final)
    |-- branche ICA   : [copie HP 1Hz pour fit] -> ICA (poids 1Hz appliqués
    |                  aux données 0.1Hz) -> drop canaux aux -> average ref
    |                  -> décimation 250Hz -> save preprocessed-ica/
    `-- branche noICA : drop canaux aux -> average ref -> décimation 250Hz
                       -> save preprocessed-noica/

Ce script prend en entrée le BIDS produit par mat_eeg_to_bids_v2.py (25 canaux,
1000 Hz, raw, référence nez).

Usage (1 job SLURM par sujet) :
    python preprocess_subject.py 5 --bids-path /path/to/dream_bids \
                                     --deriv-root /path/to/dream_bids/derivatives

────────────────────────────────────────────────────────────────────────────
CE QU'IL RESTE À FAIRE APRÈS CE SCRIPT
────────────────────────────────────────────────────────────────────────────

Pour feat_extract_umap_fooof.py (classifieur classique PSD/cov/cosp/FOOOF) :
  - Lire depuis derivatives/preprocessed-ica/ (pas preprocessed-noica/, pas
    la racine BIDS).
  - Mettre à jour config.py : SFREQ -> 250.0 (au lieu de 1000.0)
    -> N_SAMPLES = 250 * 30 = 7500 (au lieu de 30000)
    -> WINDOW = 250 (Welch 1s, au lieu de 1000)
  - HP final = 0.1Hz (matche le hardware d'origine, cf "MNE trick" dans le
    code : ICA est fit sur une copie à 1Hz mais appliquée aux données à
    0.1Hz). psd_delta reste donc comparable au pipeline Arthur sur ce point.
  - L'average reference change les amplitudes absolues -> cov/cosp seront
    différents de ceux d'Arthur (référence nez). À documenter.
  - Mettre à jour le docstring de feat_extract : la note
    "No software filtering is applied" devient fausse, les données d'entrée
    sont maintenant preprocessées (ZapLine + filtre + ICA + average ref).

Pour le futur réseau DL :
  - Entraîner et comparer sur preprocessed-ica/ ET preprocessed-noica/
    (ablation ICA vs no-ICA pour la tâche HR/LR).
  - Extraction des epochs 30s par stade (atomic stages S1/S2/S3/S4/REM),
    identique à la logique de feat_extract mais en gardant le signal brut
    (19 x 7500 samples par epoch) au lieu de features agrégées.
  - Z-score par epoch par canal (normalisation au moment du DataLoader,
    PAS sauvegardé ici -> dépend du split train/val/test).
  - Construire le vecteur de labels HR/LR par sujet (cf. SUBJECT_LIST
    d'Arthur, en attente de réponse) et le répéter pour chaque epoch du
    sujet correspondant.
  - Décider AutoReject strict ou permissif pour le DL (le réseau encaisse
    mieux les epochs légèrement bruitées que le classifieur classique).
"""

import argparse
from pathlib import Path

import mne
import mne_bids
import numpy as np
from autoreject import AutoReject

# ZapLine-plus pour le bruit de ligne 50Hz, plus robuste que notch_filter
# classique sur signal non-stationnaire (cf. dérive LF sévère s29/s31/s9)
from mne_denoise.zapline import ZapLine

from config import CH_NAMES, CH_TYPES, SFREQ


# ─── paramètres de preprocessing ──────────────────────────────────────────
# Regroupés ici pour être visibles d'un coup d'oeil et faciles à ablater.

LINE_FREQ = 50.0       # bruit de ligne secteur (France/Lyon)

# Deux cutoffs HP distincts ("MNE trick"), utilisés uniquement par la
# branche ICA :
# - HP_FREQ_ICA (1.0 Hz)   : utilisé UNIQUEMENT pour calculer les poids ICA
#                            sur une copie temporaire. ICA est statistiquement
#                            plus stable et converge mieux avec HP >= 1Hz
#                            (cf tutoriels MNE, et Kessler et al. 2025 montrent
#                            que des cutoffs HP plus élevés améliorent même le
#                            décodage : https://arxiv.org/pdf/2410.14453).
# - HP_FREQ_FINAL (0.1 Hz) : appliqué aux données réellement sauvegardées,
#                            dans LES DEUX branches. Matche le HP hardware
#                            d'origine (cf sidecar BIDS) -> préserve le delta
#                            complet (slow waves, SWS), critique pour le
#                            sommeil et pour psd_delta dans feat_extract.
HP_FREQ_ICA = 1.0
HP_FREQ_FINAL = 0.1

SFREQ_TARGET = 250.0   # décimation finale (cf SFREQ_PREPROC à ajouter dans config.py)

# 9 sujets avec saturation détectée lors de l'audit initial. Le ou les
# canaux exacts ne sont pas encore identifiés -> à inspecter manuellement
# (cf. section "inspection manuelle" plus bas) et compléter ce dict.
# Format : {sujet: [liste de canaux à marquer comme 'bads']}
KNOWN_BAD_CHANNELS = {
    5: [], 6: [], 17: [], 19: [], 20: [], 26: [], 27: [], 28: [], 37: [],
}

# Les 3 canaux utilisés comme proxy EOG/EMG pour guider l'ICA (branche ICA
# uniquement), puis jetés dans les deux branches.
ICA_AUX_CHANNELS = ['EOG_L', 'EOG_R', 'EMG_chin']

N_EEG = 19  # cf config.py : les 19 premiers canaux de CH_NAMES sont l'EEG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-path', type=Path, required=True,
                         help="Racine du BIDS produit par mat_eeg_to_bids_v2.py")
    parser.add_argument('--deriv-root', type=Path, required=True,
                         help="Racine derivatives (contiendra preprocessed-ica/ "
                              "et preprocessed-noica/)")
    return parser.parse_args()


def load_raw(bids_path, sub_str):
    """Charge le raw BIDS (25 canaux, 1000Hz, référence nez)."""
    bids_input = mne_bids.BIDSPath(
        subject=sub_str, task='sleep', root=bids_path, datatype='eeg'
    )
    raw = mne_bids.read_raw_bids(bids_input, verbose=False)
    raw.load_data()  # nécessaire pour filtrage/ICA/resample
    return raw


def mark_bad_channels(raw, sub):
    """Marque les canaux connus comme défectueux pour ce sujet.

    raw.info['bads'] est respecté par average_reference (exclus de la
    moyenne), par l'ICA (exclus du fit), et peut être interpolé plus tard
    si besoin (raw.interpolate_bads()).
    """
    bads = KNOWN_BAD_CHANNELS.get(sub, [])
    if bads:
        raw.info['bads'].extend(bads)
        print(f"  [s{sub}] bad channels marqués : {bads}")
    else:
        print(f"  [s{sub}] aucun bad channel connu "
              f"-> à vérifier manuellement si s{sub} fait partie des 9 "
              f"sujets saturés de l'audit")
    return raw


def apply_zapline(raw):
    """Retire le bruit de ligne 50Hz avec ZapLine-plus.

    Préféré au notch_filter classique : ZapLine préserve le rang du signal
    (pas de distorsion de phase comme un filtre zéro-phase classique) et son
    mode adaptatif gère le bruit de ligne chunk par chunk, ce qui est
    important sur 6-11h de signal non-stationnaire (cf. dérive LF s29/s31/s9).

    Commun aux deux branches -> fait une seule fois, avant le fork.
    """
    zapline = ZapLine(sfreq=raw.info['sfreq'], line_freq=LINE_FREQ, adaptive=True)
    raw = zapline.fit_transform(raw)
    return raw


def apply_highpass_final(raw):
    """Filtre passe-haut LÉGER (0.1Hz) appliqué aux données finales.

    Matche le HP hardware d'origine (cf sidecar BIDS 'Highpass RC filter':
    0.1Hz) -> ne retire quasiment rien de plus que ce que le hardware a
    déjà fait, préserve les ondes lentes (slow waves, delta) importantes
    pour l'analyse du sommeil et pour psd_delta dans feat_extract.

    Commun aux deux branches -> fait une seule fois, avant le fork.
    Un filtre HP plus agressif (1Hz) est appliqué séparément, sur une copie,
    uniquement pour le fit de l'ICA dans la branche ICA (cf make_ica_fit_copy).
    """
    raw.filter(l_freq=HP_FREQ_FINAL, h_freq=None, verbose=False)
    return raw


def make_ica_fit_copy(raw):
    """Crée une copie du raw filtrée à HP_FREQ_ICA (1Hz), utilisée
    UNIQUEMENT pour estimer les poids ICA (branche ICA uniquement).

    'MNE trick' : ICA est sensible aux dérives lentes et converge mal sur
    un signal HP 0.1Hz. On fit donc sur une copie plus filtrée, mais on
    applique ensuite les poids obtenus aux données réelles (HP 0.1Hz),
    qui ne sont jamais elles-mêmes filtrées à 1Hz.
    """
    raw_for_ica = raw.copy()
    raw_for_ica.filter(l_freq=HP_FREQ_ICA, h_freq=None, verbose=False)
    return raw_for_ica


def run_ica(raw, raw_for_ica):
    """ICA : poids calculés sur raw_for_ica (HP 1Hz), appliqués à raw (HP 0.1Hz).

    Fait sur le signal CONTINU (avant epoching) : plus de données -> ICA
    plus stable statistiquement que sur des epochs courts de 30s.

    On garde EOG_L/EOG_R/EMG_chin jusqu'ici uniquement pour guider la
    détection automatique des composantes à exclure. Ils sont jetés juste
    après par drop_aux_channels (commun aux deux branches).

    N'est appelé QUE dans la branche ICA -> raw passé ici est une copie
    dédiée (cf bloc fork dans __main__), donc pas d'effet sur la branche
    noICA.
    """
    # n_components=19 car on veut au final ne garder que l'info portée par
    # les 19 canaux EEG ; les 3 canaux aux ne servent qu'à la détection
    ica = mne.preprocessing.ICA(n_components=19, method='fastica',
                                 random_state=42, max_iter='auto')
    ica.fit(raw_for_ica, verbose=False)

    # détection automatique des composantes corrélées aux canaux EOG
    # (utilise raw_for_ica car find_bads_eog recalcule des scores internes
    # cohérents avec les données sur lesquelles l'ICA a été fit)
    eog_indices, eog_scores = ica.find_bads_eog(
        raw_for_ica, ch_name=['EOG_L', 'EOG_R'], verbose=False
    )

    # pas de find_bads_emg natif dans MNE pour ce cas -> détection par
    # corrélation manuelle avec EMG_chin (composantes musculaires ont
    # typiquement un spectre haute fréquence dominant)
    emg_indices, emg_scores = ica.find_bads_eog(
        raw_for_ica, ch_name='EMG_chin', threshold=3.0, verbose=False
    )
    # find_bads_eog réutilisé ici car il fait une corrélation canal/composante
    # générique ; le seuil plus permissif (3.0) reflète le fait que EMG_chin
    # n'est pas un vrai canal EOG, donc la corrélation attendue est plus faible

    bad_components = sorted(set(eog_indices) | set(emg_indices))
    ica.exclude = bad_components
    print(f"  composantes ICA exclues (EOG+EMG) : {bad_components}")

    # application des poids (fit sur raw_for_ica à 1Hz) aux données
    # réelles raw (à 0.1Hz) -> "best of both worlds"
    raw = ica.apply(raw, verbose=False)
    return raw, ica


def drop_aux_channels(raw):
    """Retire EOG_L/EOG_R/EMG_chin/misc* : ne garde que les 19 EEG.

    Commun aux deux branches. Dans la branche ICA, ces canaux ont servi à
    guider l'ICA (cf run_ica) ; dans la branche noICA ils n'ont servi à
    rien et sont simplement absents du résultat. Dans les deux cas ils ne
    font pas partie des features ni de l'input réseau.
    """
    eeg_channels = CH_NAMES[:N_EEG]
    raw.pick(eeg_channels)
    return raw


def apply_average_reference(raw):
    """Re-référencement average reference.

    Commun aux deux branches. Remplace la référence nez d'origine. Les
    canaux marqués 'bads' sont automatiquement exclus du calcul de la
    moyenne par MNE.
    Conséquence : cov/cosp dans feat_extract seront différents de ceux
    d'Arthur (qui utilise la référence nez d'origine) -> à documenter.
    """
    raw.set_eeg_reference('average', verbose=False)
    return raw


def apply_decimation(raw):
    """Décimation 1000 -> 250 Hz.

    Commun aux deux branches. Fait APRÈS l'ICA dans la branche ICA (qui
    bénéficie de la résolution temporelle complète) et APRÈS le filtrage
    (qui doit être fait avant un resample pour éviter l'aliasing). 250Hz
    couvre largement FREQ_DICT (max 35Hz) et FOOOF_FREQ_RANGE (max 45Hz)
    du config.py existant.
    """
    raw.resample(SFREQ_TARGET, verbose=False)
    return raw


def save_bids_derivatives(raw, sub_str, deriv_root, branch_name):
    """Sauvegarde en BIDS derivatives, format BrainVision.

    branch_name : 'preprocessed-ica' ou 'preprocessed-noica' -> deux
    dossiers derivatives séparés sous deriv_root.

    BrainVision choisi pour rester cohérent avec feat_extract_umap_fooof.py
    qui lit déjà du BrainVision depuis le BIDS d'origine. Les annotations
    (hypnogrammes per/jbe) sont préservées par raw.set_annotations en amont
    (déjà présentes dans le raw chargé) et ré-écrites automatiquement par
    write_raw_bids.
    """
    deriv_path = deriv_root / branch_name
    deriv_output = mne_bids.BIDSPath(
        subject=sub_str, task='sleep', root=deriv_path,
        datatype='eeg', processing='clean'  # entité BIDS standard pour derivatives
    )
    mne_bids.write_raw_bids(raw, deriv_output, overwrite=True,
                             allow_preload=True, format='BrainVision')
    return deriv_output


if __name__ == '__main__':
    args = parse_args()
    sub = args.subject
    sub_str = str(sub).zfill(2)

    print(f"=== Preprocessing sujet {sub} ===")

    # ─── tronc commun ────────────────────────────────────────────────────
    raw = load_raw(args.bids_path, sub_str)
    raw = mark_bad_channels(raw, sub)
    raw = apply_zapline(raw)
    raw = apply_highpass_final(raw)  # HP 0.1Hz, partagé par les deux branches

    # ─── fork : une copie indépendante par branche à partir d'ici ────────
    raw_ica_branch = raw.copy()
    raw_noica_branch = raw.copy()

    # --- branche ICA (pour feat_extract) ---
    print("  -- branche ICA --")
    raw_for_ica = make_ica_fit_copy(raw_ica_branch)   # copie HP 1Hz, fit seulement
    raw_ica_branch, ica = run_ica(raw_ica_branch, raw_for_ica)
    raw_ica_branch = drop_aux_channels(raw_ica_branch)
    raw_ica_branch = apply_average_reference(raw_ica_branch)
    raw_ica_branch = apply_decimation(raw_ica_branch)
    out_ica = save_bids_derivatives(raw_ica_branch, sub_str, args.deriv_root,
                                     branch_name='preprocessed-ica')
    print(f"  Done (ICA)   : {out_ica}")

    # --- branche noICA (pour ablation DL) ---
    print("  -- branche noICA --")
    raw_noica_branch = drop_aux_channels(raw_noica_branch)
    raw_noica_branch = apply_average_reference(raw_noica_branch)
    raw_noica_branch = apply_decimation(raw_noica_branch)
    out_noica = save_bids_derivatives(raw_noica_branch, sub_str, args.deriv_root,
                                       branch_name='preprocessed-noica')
    print(f"  Done (noICA) : {out_noica}")

    # ─── ce qui n'est PAS fait ici, volontairement ──────────────────────
    # - epoching 30s par stade : reste dans feat_extract / futur dataset DL
    #   (les annotations per/jbe sont préservées dans les deux sorties, donc
    #   l'epoching peut se faire à partir d'ici sans recharger le BIDS brut)
    # - AutoReject : appliqué en aval, séparément pour feat_extract
    #   (strict) et pour le réseau DL (potentiellement plus permissif)
    # - z-score : dépend du split train/val/test, donc fait au moment du
    #   DataLoader, jamais ici
