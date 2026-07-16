"""Prétraitement des enregistrements EEG de sommeil.

Lit le BIDS produit par mat_eeg_to_bids (25 canaux, 1000 Hz, référence nez) et
écrit un derivative par branche sous derivatives/preprocessed-<branche>/.

Trois branches : noica, ica, iclabel. L'objet ICA de chaque branche est sauvé
dans derivatives/ica/ pour inspection offline (inspect_ica.py).

Étapes, dans l'ordre :

  1. Notch 50/100 Hz                     [commun]
  2. Passe-haut 0.1 Hz                   [commun]
     |-- 3a. Copie filtrée à 1 Hz        [ica, iclabel]
     |   3b. ICA fit + apply             [ica, iclabel]
     |   3c. Sauvegarde de l'ICA         [ica, iclabel]
  4. Drop des canaux auxiliaires         [commun]  ne garde que les 19 EEG
  5. Décimation                          [commun]  no-op tant que DECIMATE=False
  6. Sauvegarde BrainVision              [commun]  un fichier par branche

Le choix des branches, de la référence d'enregistrement et des fréquences de
coupure est justifié dans le README (section Choix méthodologiques).

Note sujets 21/22 : prétraités normalement, exclus en aval par classify.py
(EXCLUDED_SUBJECTS dans config.py). Les données peuvent servir à d'autres
analyses.
"""

import argparse
from pathlib import Path

import mne
import mne_bids
import numpy as np
from mne_icalabel import label_components

from .config import (
    CH_NAMES, SFREQ,
    LINE_FREQ, HP_FREQ_FINAL, HP_FREQ_ICA, SFREQ_TARGET, DECIMATE, N_EEG,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids-path', type=Path, required=True,
                        help="Racine du BIDS produit par mat_eeg_to_bids.py")
    parser.add_argument('--deriv-root', type=Path, required=True,
                        help="Racine derivatives (contiendra preprocessed-ica/, "
                             "preprocessed-noica/, ica/)")
    parser.add_argument('--branches', nargs='+',
                        choices=['ica', 'noica', 'iclabel'],
                        default=['ica', 'noica', 'iclabel'],
                        help="Branches a executer. Ex: --branches ica")
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


def apply_notch(raw: mne.io.BaseRaw) -> None:
    """1. Retire le bruit de ligne 50Hz (+ harmonique 100Hz) par notch filter.

    LINE_FREQ=50 et son harmonique 100Hz. Le notch s'applique avant la
    décimation (raw encore à 1000Hz) . EOG/EMG portent aussi la raie secteur et sont
    filtrés ici, puis droppés à l'étape 4 -> évite de polluer la détection ICA
    des composantes oculaires qui suit.

    Commun aux deux branches -> fait une seule fois, avant le fork.
    """
    raw.notch_filter(
        [LINE_FREQ, 2 * LINE_FREQ],                 # 50 et 100 Hz
        filter_length='auto',
        phase='zero',
        verbose=False,
    )


def apply_highpass_final(raw: mne.io.BaseRaw) -> None:
    """2. Filtre passe-haut 0.1Hz appliqué aux données finales.

    Matche le HP hardware d'origine (sidecar BIDS : 'Highpass RC filter'
    0.1Hz) -> ne retire quasiment rien de plus que le hardware.

    Commun aux deux branches -> fait une seule fois, avant le fork.
    """
    raw.filter(l_freq=HP_FREQ_FINAL, h_freq=None, verbose=False)


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
    mouvements verticaux), et leur différence EOG_L-EOG_R l'oculaire horizontal
    (saccades latérales). EMG_chin sert à détecter les composantes musculaires.

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

        method='picard',       # reco doc MNE pour l'EEG réel : plus robuste que
                               # FastICA quand les sources ne sont pas parfaitement
                               # indépendantes (cas typique EEG), converge + vite

        random_state=42,       # reproductibilité du fit ICA
        max_iter='auto',
    )

    # fit sur la copie HP 1Hz pour stabilité
    ica.fit(raw_for_ica, verbose=False)

    # Détection des composantes oculaires via EOG_L, EOG_R et une voie horizontale.
    # L'EOG horizontal (saccades latérales du REM) forme un dipôle gauche-droite.
    # La soustraction (EOG_L - EOG_R) double l'amplitude de ce dipôle hors-phase
    # et élimine par rejet de mode commun (CMR) le bruit EEG frontal en-phase.
    # Création sur copie locale pour détection uniquement (non conservée dans le raw).
    raw_eog = mne.set_bipolar_reference(
        raw_for_ica, anode='EOG_L', cathode='EOG_R',
        ch_name='EOG_horiz', drop_refs=False, copy=True, verbose=False,
    )
    raw_eog.set_channel_types({'EOG_horiz': 'eog'}, verbose=False)

    # Méthode retenue : corrélation absolue, seuil 0.6.
    # Analyse empirique sur 38 sujets (analyze_thresholds.py + analyze_zscore.py)

    # La corrélation 0.6 donne ~1.4 rejets/sujet (plage physiologique 1-2),
    # CV inter-sujets stable (~0.6), et préserve le signal frontal (objectif
    # mécanistique : interpréter la zone biologique HR vs LR).
    
    eog_indices, eog_scores = ica.find_bads_eog(
        raw_eog, ch_name=['EOG_L', 'EOG_R', 'EOG_horiz'],
        measure='correlation', threshold=0.6, verbose=False,
    )

    # find_bads_muscle retiré définitivement.
    # Raisons empiriques : balayage seuils 0.1-0.9 sur 38 sujets => CV entre
    # 0.77 et 1.13 sans zone stable, distribution quasi-uniforme en log (pas
    # de frontière bruit/artefact). Atonie musculaire du sommeil = peu de
    # vrai EMG à capturer. La branche noica couvre le risque résiduel.
    
    ica.exclude = list(eog_indices)
    print(f"  composantes ICA exclues (EOG corr 0.6) : {ica.exclude}")

    # application des poids (calculés sur copie 1Hz) aux données réelles (0.1Hz)
    # -> composantes EOG/EMG retirées, ondes lentes préservées
    raw = ica.apply(raw, verbose=False)
    return raw, ica, eog_indices, eog_scores


# seuil de rejet ICLabel : proba > 0.9 pour eye/muscle. Conservateur, justifié
# pour le sommeil : (a) atonie musculaire -> peu d'EMG attendu ; (b) ICLabel est
# entraîné sur de l'EEG éveillé, donc risque de faux positifs sur les ondes
# lentes (delta/fuseaux non représentés à l'entraînement) -> on ne rejette que
# ce dont ICLabel est très sûr. Littérature : seuils 0.8-0.9 quasi équivalents.
ICLABEL_THRESHOLD = 0.9
ICLABEL_REJECT_LABELS = ('eye blink', 'muscle artifact')


def run_ica_iclabel(
    raw: mne.io.BaseRaw,
    raw_for_ica: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA]:
    """3b'. Variante ICLabel : ICA Picard-extended + labellisation automatique.

    Branche indépendante de run_ica (FastICA->Picard + find_bads). Sert de
    méthode de rejet alternative pour comparaison empirique en aval.

    ICLabel impose 3 prérequis (cf doc mne-icalabel), appliqués sur une copie
    dédiée à la labellisation pour ne pas altérer la décomposition :
      - décomposition extended Infomax : obtenue via Picard extended
        (method='picard', ortho=False, extended=True) -> décomposition
        identique à extended Infomax mais convergence + rapide (reco MNE-BIDS).
      - référence average (CAR)
      - filtre 1-100 Hz
    Rejet : composantes labellisées eye/muscle avec proba > ICLABEL_THRESHOLD.
    """
    ica = mne.preprocessing.ICA(
        n_components=0.99,
        method='picard',
        fit_params=dict(ortho=False, extended=True),  # = extended Infomax (ICLabel)
        random_state=42,
        max_iter='auto',
    )
    ica.fit(raw_for_ica, verbose=False)

    # copie dédiée à la labellisation : ICLabel exige CAR + filtre 1-100Hz.
    # raw_for_ica est déjà HP 1Hz -> on ajoute seulement le LP 100Hz et la CAR.
    raw_label = raw_for_ica.copy().pick('eeg')
    raw_label.filter(l_freq=None, h_freq=100.0, verbose=False)
    raw_label.set_eeg_reference('average', verbose=False)

    labels_dict = label_components(raw_label, ica, method='iclabel')
    labels = labels_dict['labels']
    probas = labels_dict['y_pred_proba']

    ica.exclude = [
        i for i, (lab, p) in enumerate(zip(labels, probas))
        if lab in ICLABEL_REJECT_LABELS and p > ICLABEL_THRESHOLD
    ]
    print(f"  composantes ICLabel exclues (eye/muscle p>{ICLABEL_THRESHOLD}) : "
          f"{ica.exclude}")

    raw = ica.apply(raw, verbose=False)
    return raw, ica


def save_ica(
    ica: mne.preprocessing.ICA,
    sub_str: str,
    deriv_root: Path,
    suffix: str = '',
) -> Path:
    """3c. Sauvegarde de l'objet ICA pour inspection offline.

    Stocké dans derivatives/ica/ (hors des branches preprocessed-*) pour
    permettre de relancer inspect_ica.py sans refaire le preprocessing.
    suffix distingue les ICA des différentes branches (ex: '-iclabel') :
    sans suffixe = branche ICA principale (Picard + find_bads).

    Format .fif (natif MNE) : rechargeable avec mne.preprocessing.read_ica().
    Nommage BIDS-inspired : sub-XX_task-sleep{suffix}_ica.fif
    """
    ica_dir = deriv_root / "ica"
    ica_dir.mkdir(parents=True, exist_ok=True)
    ica_path = ica_dir / f"sub-{sub_str}_task-sleep{suffix}_ica.fif"
    ica.save(ica_path, overwrite=True)
    return ica_path


def drop_aux_channels(raw: mne.io.BaseRaw) -> None:
    """4. Retire EOG_L/EOG_R/EMG_chin/misc* -> 19 canaux EEG uniquement.

    Commun aux deux branches. Ces canaux ont servi à guider l'ICA dans la
    branche ICA ; dans la branche noICA ils n'ont servi à rien. Dans les
    deux cas ils ne font pas partie des features ni de l'input réseau.

    Aucun re-référencement n'est appliqué en aval (cf Note référence en
    en-tête) : les 19 EEG conservent la référence nez d'enregistrement.
    """
    raw.pick(CH_NAMES[:N_EEG])


def apply_decimation(raw: mne.io.BaseRaw) -> None:
    """6. Décimation optionnelle 1000 -> SFREQ_TARGET Hz, pilotée par DECIMATE (config.py).

    DECIMATE=False (défaut actuel, 03/07/2026) : réplication exacte thèse Arthur
    §1.2.3, données gardées à 1000Hz -> no-op.
    DECIMATE=True : fait APRÈS l'ICA (résolution temporelle complète pour le fit)
    et APRÈS le filtrage HP (anti-aliasing implicite, signal déjà bandlimité).
    SFREQ_TARGET=250Hz couvre largement FREQ_DICT max 35Hz et FOOOF_FREQ_RANGE
    max 45Hz. Réduit taille fichiers et temps de calcul par ~4.

    Si DECIMATE=True, il faut aussi ajuster WINDOW dans config.py (actuellement
    câblé pour 1000Hz) -> non fait automatiquement, à vérifier manuellement.
    """
    if DECIMATE:
        raw.resample(SFREQ_TARGET, verbose=False)
    expected_sfreq = SFREQ_TARGET if DECIMATE else SFREQ
    assert raw.info["sfreq"] == expected_sfreq, (
        f"apply_decimation: sfreq={raw.info['sfreq']} != attendu {expected_sfreq} "
        f"(DECIMATE={DECIMATE}) -> incohérence config.py, à corriger avant de continuer."
    )


def save_bids_derivatives(
    raw: mne.io.BaseRaw,
    sub_str: str,
    deriv_root: Path,
    branch_name: str,
) -> mne_bids.BIDSPath:
    """7. Sauvegarde en BIDS derivatives, format BrainVision.

    branch_name : 'preprocessed-ica' ou 'preprocessed-noica'
    -> deux dossiers séparés sous deriv_root.

    Format BrainVision pour cohérence avec feat_extract_umap_fooof
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

def main():
    args    = parse_args()
    sub     = args.subject
    sub_str = str(sub).zfill(2)

    print(f"=== Preprocessing sujet {sub} ===")

    # sujets 21 et 22 : preprocessés normalement (pas de sys.exit ici).
    # L'exclusion de l'analyse HR/LR se fait en aval dans classify
    # via EXCLUDED_SUBJECTS dans config

    # ── tronc commun (étapes 1-2) ────────────────────────────────────────────
    raw = load_raw(args.bids_path, sub_str)
    apply_notch(raw)                   # 1. notch 50Hz (+100Hz)
    apply_highpass_final(raw)          # 2. HP 0.1Hz

    # ── fork : copie indépendante par branche ────────────────────────────────
    raw_ica_branch     = raw.copy()
    raw_noica_branch   = raw.copy()
    raw_iclabel_branch = raw.copy()

    # ── branche ICA (étapes 3a-3c + 4-7) -> feat_extract + classifieur ──────
    if 'ica' in args.branches:
     print("  -- branche ICA --")
     raw_for_ica         = make_ica_fit_copy(raw_ica_branch)        # 3a. copie HP 1Hz
     raw_ica_branch, ica, _eog_idx, _eog_sc = run_ica(raw_ica_branch, raw_for_ica)  # 3b. ICA
     ica_path            = save_ica(ica, sub_str, args.deriv_root)  # 3c. save ICA
     print(f"  ICA sauvegardé  : {ica_path}")

     # log CSV des scores EOG par composante (corr absolue, seuil 0.6).
     # find_bads_muscle retiré (23 juin 2026, voir run_ica).
     import csv as _csv
     ica_scores_log = args.deriv_root / "ica" / "ica_rejection_scores.csv"
     write_header = not ica_scores_log.exists()
     with open(ica_scores_log, 'a', newline='') as _f:
         _w = _csv.writer(_f)
         if write_header:
             _w.writerow(['subject', 'comp', 'eog_corr', 'eog_rejected'])
         for _i in range(ica.n_components_):
             _w.writerow([
                 sub_str, _i,
                 round(float(np.abs(np.atleast_2d(_eog_sc))[:, _i].max()) if _i < np.atleast_2d(_eog_sc).shape[1] else float('nan'), 6),
                 int(_i in _eog_idx),
             ])
     drop_aux_channels(raw_ica_branch)                              # 4. drop aux
     apply_decimation(raw_ica_branch)                               # 6. no-op si DECIMATE=False
     out_ica = save_bids_derivatives(                               # 7. save
         raw_ica_branch, sub_str, args.deriv_root, 'preprocessed-ica'
     )
     print(f"  Done (ICA)      : {out_ica}")

    # ── branche noICA (étapes 4-7 uniquement) -> ablation DL ────────────────
    if 'noica' in args.branches:
     print("  -- branche noICA --")
     drop_aux_channels(raw_noica_branch)                            # 4. drop aux
     apply_decimation(raw_noica_branch)                             # 6. no-op si DECIMATE=False
     out_noica = save_bids_derivatives(                             # 7. save
         raw_noica_branch, sub_str, args.deriv_root, 'preprocessed-noica'
     )
     print(f"  Done (noICA)    : {out_noica}")

    # ── branche ICLabel (étapes 3a-3c' + 4-7) -> comparaison rejet alternatif ─
    # ICA Picard-extended + labellisation ICLabel (≠ branche ica : Picard +
    # find_bads). ICA distincte, fittée séparément, sauvée sous suffixe -iclabel.
    if 'iclabel' in args.branches:
     print("  -- branche ICLabel --")
     raw_for_iclabel             = make_ica_fit_copy(raw_iclabel_branch)   # 3a. copie HP 1Hz
     raw_iclabel_branch, ica_icl = run_ica_iclabel(                        # 3b'. ICA+ICLabel
         raw_iclabel_branch, raw_for_iclabel
     )
     icl_path = save_ica(ica_icl, sub_str, args.deriv_root, suffix='-iclabel')  # 3c'
     print(f"  ICA sauvegardé  : {icl_path}")
     drop_aux_channels(raw_iclabel_branch)                                 # 4. drop aux
     apply_decimation(raw_iclabel_branch)                                  # 6. no-op si DECIMATE=False
     out_iclabel = save_bids_derivatives(                                  # 7. save
         raw_iclabel_branch, sub_str, args.deriv_root, 'preprocessed-iclabel'
     )
     print(f"  Done (ICLabel)  : {out_iclabel}")


if __name__ == "__main__":
    main()
