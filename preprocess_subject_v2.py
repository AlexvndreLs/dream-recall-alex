"""
Preprocessing offline des données EEG sommeil (dataset Ruby/Eichenlaub, CRNL Lyon)
avant extraction de features (feat_extract_umap_fooof_v2.py) et entraînement réseau DL.

Ce script produit DEUX sorties BIDS derivatives par sujet :

  - derivatives/preprocessed-ica/   : pipeline complet AVEC ICA
                                       -> destiné à feat_extract (PSD/cov/cosp)
                                          et classifieur LDA/Riemannian
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

Pipeline (ordre important, cf justifications inline) :

  1. Notch 50/100Hz           [commun]  retirer bruit de ligne secteur
  2. HP filter 0.1Hz           [commun]  retirer dérive DC, préserver delta
     |
     |-- branche ICA :
  3a.    Copie HP 1Hz          [ICA]     stabiliser le fit ICA (MNE trick)
  3b.    ICA fit + apply       [ICA]     retirer composantes EOG/EMG
  3c.    Sauvegarde ICA        [ICA]     pour inspection offline (inspect_ica.py)
     |
  4. Drop canaux aux           [commun]  ne garder que les 19 EEG
  5. Average reference         [commun]  rereferencer, remplace ref nez
  6. Décimation 250Hz          [commun]  réduire volume, suffisant pour <=45Hz
  7. Save BrainVision          [commun]  un fichier par branche par sujet

Ce script prend en entrée le BIDS produit par mat_eeg_to_bids_v2.py (25 canaux,
1000 Hz, raw, référence nez).

Usage (1 job SLURM par sujet) :
    python preprocess_subject_v2.py 5 --bids-path /path/to/dream_bids \\
                                       --deriv-root /path/to/dream_bids/derivatives

CE QU'IL RESTE À FAIRE APRÈS CE SCRIPT :

Pour feat_extract_umap_fooof_v2.py (classifieur classique PSD/cov/cosp/FOOOF) :
  - Lire depuis derivatives/preprocessed-ica/
  - config_v3.py déjà mis à jour : SFREQ_PREPROC=250, N_SAMPLES=7500, WINDOW=250
  - HP final = 0.1Hz -> psd_delta comparable au pipeline Arthur sur ce point
  - L'average reference change cov/cosp vs Arthur (référence nez) -> documenter
  - AutoReject strict sur les epochs 30s avant calcul des features

Pour le futur réseau DL :
  - Entraîner et comparer sur preprocessed-ica/ ET preprocessed-noica/
  - Extraction epochs 30s par stade (19 x 7500 samples par epoch à 250Hz)
  - Normalisation robuste MAD par fenêtre au moment du DataLoader (pas ici)
    -> global (préserve topographie), robuste (résistant aux pics résiduels),
       clip ±8sigma
  - Labels HR/LR par sujet (cf SUBJECT_LIST Arthur, en attente de réponse),
    répétés pour chaque epoch du sujet
  - AutoReject permissif ou absent pour le DL (le réseau encaisse mieux les
    epochs légèrement bruitées)

Note bad channels : pas de détection/rejet de canaux entiers dans ce script.
La saturation sur les 9 sujets flaggés (s5,s6,s17,s19,s20,s26,s27,s28,s37)
est ponctuelle (quelques minutes) et non structurelle -> traitée en aval
par AutoReject au niveau epoch (rejet ou réparation de la fenêtre 30s
concernée), pas au niveau canal.

Note référence : l'average reference (étape 5) remplace la référence nez
d'origine. Cela rend cov/cosp non comparables à ceux d'Arthur (référence nez).
A documenter lors de la comparaison avec ses résultats.

Note sujets 21/22 : ces sujets sont preprocessés normalement ici.
Leur exclusion de l'analyse HR/LR se fait en aval dans classify.py
(EXCLUDED_SUBJECTS dans config_v3.py). Les données preprocessées sont
produites car elles pourront servir pour d'autres analyses (ex: réseau DL
en mode non-supervisé) ou si la raison de l'exclusion est clarifiée.
"""

import argparse
from pathlib import Path

import mne
import mne_bids
import numpy as np

from config_v3 import (
    CH_NAMES, SFREQ,
    LINE_FREQ, HP_FREQ_FINAL, HP_FREQ_ICA, SFREQ_TARGET, N_EEG,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-path', type=Path, required=True,
                        help="Racine du BIDS produit par mat_eeg_to_bids_v2.py")
    parser.add_argument('--deriv-root', type=Path, required=True,
                        help="Racine derivatives (contiendra preprocessed-ica/, "
                             "preprocessed-noica/, ica/)")
    return parser.parse_args()


# ─── fonctions ────────────────────────────────────────────────────────────────

def load_raw(bids_path: Path, sub_str: str) -> mne.io.BaseRaw:
    """Charge le raw BIDS (25 canaux, 1000Hz, référence nez).

    Les annotations hypnogrammes (per/jbe) sont lues automatiquement
    par read_raw_bids et préservées jusqu'à la sauvegarde finale ->
    epoching par stade possible en aval sans recharger le BIDS brut.
    """
    bids_input = mne_bids.BIDSPath(
        subject=sub_str, task='sleep', root=bids_path, datatype='eeg'
    )
    raw = mne_bids.read_raw_bids(bids_input, verbose=False)
    raw.load_data()  # nécessaire pour filtrage / ICA / resample
    return raw


def apply_notch(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """1. Retire le bruit de ligne 50Hz (+ harmonique 100Hz) par notch filter.

    ZapLine (meegkit.dss.dss_line) testé d'abord mais retirait 0% de la raie
    50Hz pourtant énorme (ratio 50Hz/voisins ~18 sur s5) -> remplacé par un
    notch MNE simple et robuste. Acceptable ici car les features s'arrêtent à
    45Hz (FOOOF) / 35Hz (bandes) : le signal cérébral à 50Hz n'est pas analysé,
    donc la préservation spatiale de ZapLine n'apporte rien.

    LINE_FREQ=50 et son harmonique 100Hz. Le notch s'applique avant la
    décimation (raw encore à 1000Hz, Nyquist 500Hz) : 50 et 100Hz passent
    largement. Pas de picks : EOG/EMG portent aussi la raie secteur et sont
    filtrés ici, puis droppés à l'étape 4 -> évite de polluer la détection ICA
    des composantes oculaires qui suit. phase='zero' : pas de déphasage.
    Commun aux deux branches -> fait une seule fois, avant le fork.
    """
    raw.notch_filter(
        [LINE_FREQ, 2 * LINE_FREQ],                 # 50 et 100 Hz
        filter_length='auto',
        phase='zero',
        verbose=False,
    )
    return raw


def apply_highpass_final(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """2. Filtre passe-haut 0.1Hz appliqué aux données finales.

    Matche le HP hardware d'origine (sidecar BIDS : 'Highpass RC filter'
    0.1Hz) -> ne retire quasiment rien de plus que le hardware, préserve
    les ondes delta/slow waves (0.5-4Hz) indispensables pour SWS et pour
    psd_delta dans feat_extract.

    Commun aux deux branches -> fait une seule fois, avant le fork.
    """
    raw.filter(l_freq=HP_FREQ_FINAL, h_freq=None, verbose=False)
    return raw


def make_ica_fit_copy(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """3a. Copie HP 1Hz pour le fit ICA uniquement (branche ICA).

    ICA est sensible aux dérives basse fréquence et converge mal sur un
    signal HP 0.1Hz. Cette copie filtrée à 1Hz sert uniquement à estimer
    les poids ICA -> jamais sauvegardée, jetée après run_ica.
    """
    raw_for_ica = raw.copy()
    raw_for_ica.filter(l_freq=HP_FREQ_ICA, h_freq=None, verbose=False)
    return raw_for_ica


def run_ica(
    raw: mne.io.BaseRaw,
    raw_for_ica: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA]:
    """3b. ICA : poids calculés sur la copie HP 1Hz, appliqués au raw HP 0.1Hz.

    Fait sur le signal CONTINU (avant epoching) : plus de données ->
    décomposition ICA plus stable que sur des epochs courts de 30s.

    EOG_L/EOG_R servent à détecter les composantes oculaires (clignements,
    mouvements). EMG_chin sert à détecter les composantes musculaires.
    Ces 3 canaux sont conservés jusqu'ici pour cette raison, puis jetés
    dans drop_aux_channels (commun aux deux branches).

    N'est appelé QUE dans la branche ICA. raw_ica_branch est une copie
    indépendante (cf fork dans __main__) -> pas d'effet sur raw_noica_branch.
    """
    ica = mne.preprocessing.ICA(
        n_components=0.99,     # capture 99% de la variance -> MNE choisit le
                               # nombre de composantes automatiquement.
                               # évite les instabilités numériques des
                               # composantes à variance quasi nulle (vs None)
        method='fastica',
        random_state=42,       # reproductibilité du fit ICA
        max_iter='auto',
    )

    # fit sur la copie HP 1Hz pour stabilité
    ica.fit(raw_for_ica, verbose=False)

    # détection composantes oculaires via EOG_L et EOG_R
    eog_indices, _ = ica.find_bads_eog(
        raw_for_ica, ch_name=['EOG_L', 'EOG_R'], verbose=False
    )

    # détection composantes musculaires via find_bads_muscle
    # (plus correct que find_bads_eog sur EMG_chin : détecte les composantes
    # avec spectre dominé par les hautes fréquences, sans besoin de canal
    # de référence)
    # threshold=0.5 : valeur par défaut MNE (cf. mne.preprocessing.ICA.find_bads_muscle).
    # Non validée empiriquement sur ce dataset (sommeil, sujet allongé immobile).
    # À calibrer via inspect_ica.py avant de figer ce paramètre.
    emg_indices, _ = ica.find_bads_muscle(raw_for_ica, threshold=0.5) #pourquoi 0.5 ?? a voir

    ica.exclude = sorted(set(eog_indices) | set(emg_indices))
    print(f"  composantes ICA exclues (EOG+EMG) : {ica.exclude}")

    # application des poids (calculés sur copie 1Hz) aux données réelles (0.1Hz)
    # -> composantes EOG/EMG retirées, ondes lentes préservées
    raw = ica.apply(raw, verbose=False)
    return raw, ica


def save_ica(
    ica: mne.preprocessing.ICA,
    sub_str: str,
    deriv_root: Path,
) -> Path:
    """3c. Sauvegarde de l'objet ICA pour inspection offline.

    Stocké dans derivatives/ica/ (hors des branches preprocessed-ica/noica)
    car l'objet ICA est commun aux deux : les poids sont calculés une seule
    fois et appliqués uniquement dans la branche ICA. Permet de relancer
    inspect_ica.py sans refaire le preprocessing complet.

    Format .fif (natif MNE) : rechargeable avec mne.preprocessing.read_ica().
    Nommage BIDS-inspired : sub-XX_task-sleep_ica.fif
    """
    ica_dir = deriv_root / "ica"
    ica_dir.mkdir(parents=True, exist_ok=True)
    ica_path = ica_dir / f"sub-{sub_str}_task-sleep_ica.fif"
    ica.save(ica_path, overwrite=True)
    return ica_path


def drop_aux_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """4. Retire EOG_L/EOG_R/EMG_chin/misc* -> 19 canaux EEG uniquement.

    Commun aux deux branches. Ces canaux ont servi à guider l'ICA dans la
    branche ICA ; dans la branche noICA ils n'ont servi à rien. Dans les
    deux cas ils ne font pas partie des features ni de l'input réseau.

    Drop fait AVANT average reference : la moyenne ne doit porter que
    sur les 19 EEG, pas sur les canaux auxiliaires.
    """
    raw.pick(CH_NAMES[:N_EEG])
    return raw


def apply_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """5. Re-référencement average reference sur les 19 EEG.

    Commun aux deux branches. Remplace la référence nez d'origine
    (cf sidecar BIDS : EEGReference = 'tip of the nose').
    Conséquence documentée : cov/cosp dans feat_extract seront différents
    de ceux d'Arthur qui utilise la référence nez -> à mentionner lors de
    la comparaison avec ses résultats publiés.   !!!
    """
    raw.set_eeg_reference('average', verbose=False)
    return raw


def apply_decimation(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """6. Décimation 1000 -> 250 Hz.

    Commun aux deux branches. Fait APRÈS l'ICA (qui bénéficie de la
    résolution temporelle complète à 1000Hz) et APRÈS le filtrage HP
    (anti-aliasing implicite : le signal est déjà bandlimité). 250Hz
    couvre largement FREQ_DICT max 35Hz et FOOOF_FREQ_RANGE max 45Hz.
    Réduit la taille des fichiers et le temps de calcul par un facteur 4.
    """
    raw.resample(SFREQ_TARGET, verbose=False)
    return raw


def save_bids_derivatives(
    raw: mne.io.BaseRaw,
    sub_str: str,
    deriv_root: Path,
    branch_name: str,
) -> mne_bids.BIDSPath:
    """7. Sauvegarde en BIDS derivatives, format BrainVision.

    branch_name : 'preprocessed-ica' ou 'preprocessed-noica'
    -> deux dossiers séparés sous deriv_root.

    Format BrainVision pour cohérence avec feat_extract_umap_fooof_v2.py
    qui lit déjà du BrainVision depuis le BIDS d'origine.
    Les annotations hypnogrammes (per/jbe) sont ré-écrites automatiquement
    par write_raw_bids car elles sont portées par raw.annotations.
    """
    deriv_output = mne_bids.BIDSPath(
        subject=sub_str, task='sleep',
        root=deriv_root / branch_name,
        datatype='eeg',
        processing='clean',  # entité BIDS standard pour les derivatives
    )
    mne_bids.write_raw_bids(
        raw, deriv_output,
        overwrite=True, allow_preload=True, format='BrainVision',
    )
    return deriv_output


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args    = parse_args()
    sub     = args.subject
    sub_str = str(sub).zfill(2)

    print(f"=== Preprocessing sujet {sub} ===")

    # sujets 21 et 22 : preprocessés normalement (pas de sys.exit ici).
    # L'exclusion de l'analyse HR/LR se fait en aval dans classify.py
    # via EXCLUDED_SUBJECTS dans config_v3.py.

    # ── tronc commun (étapes 1-2) ────────────────────────────────────────────
    raw = load_raw(args.bids_path, sub_str)
    raw = apply_notch(raw)             # 1. notch 50Hz (+100Hz)
    raw = apply_highpass_final(raw)    # 2. HP 0.1Hz

    # ── fork : copie indépendante par branche ────────────────────────────────
    raw_ica_branch   = raw.copy()
    raw_noica_branch = raw.copy()

    # ── branche ICA (étapes 3a-3c + 4-7) -> feat_extract + classifieur ──────
    print("  -- branche ICA --")
    raw_for_ica         = make_ica_fit_copy(raw_ica_branch)        # 3a. copie HP 1Hz
    raw_ica_branch, ica = run_ica(raw_ica_branch, raw_for_ica)     # 3b. ICA
    ica_path            = save_ica(ica, sub_str, args.deriv_root)  # 3c. save ICA
    print(f"  ICA sauvegardé  : {ica_path}")
    raw_ica_branch      = drop_aux_channels(raw_ica_branch)        # 4. drop aux
    raw_ica_branch      = apply_average_reference(raw_ica_branch)  # 5. avg ref
    raw_ica_branch      = apply_decimation(raw_ica_branch)         # 6. 250Hz
    out_ica = save_bids_derivatives(                               # 7. save
        raw_ica_branch, sub_str, args.deriv_root, 'preprocessed-ica'
    )
    print(f"  Done (ICA)      : {out_ica}")

    # ── branche noICA (étapes 4-7 uniquement) -> ablation DL ────────────────
    print("  -- branche noICA --")
    raw_noica_branch = drop_aux_channels(raw_noica_branch)         # 4. drop aux
    raw_noica_branch = apply_average_reference(raw_noica_branch)   # 5. avg ref
    raw_noica_branch = apply_decimation(raw_noica_branch)          # 6. 250Hz
    out_noica = save_bids_derivatives(                             # 7. save
        raw_noica_branch, sub_str, args.deriv_root, 'preprocessed-noica'
    )
    print(f"  Done (noICA)    : {out_noica}")

    # ── ce qui n'est PAS fait ici, volontairement ────────────────────────────
    # - bad channels : saturation ponctuelle sur 9 sujets traitée en aval
    #   par AutoReject au niveau epoch, pas au niveau canal (cf docstring)
    # - epoching 30s par stade : fait en aval dans feat_extract / dataset DL
    #   (annotations per/jbe préservées dans les deux sorties)
    # - AutoReject : appliqué en aval, strict pour feat_extract,
    #   permissif ou absent pour le réseau DL
    # - normalisation MAD : faite au moment du DataLoader DL, jamais ici
    #   (dépend du split train/val/test)
