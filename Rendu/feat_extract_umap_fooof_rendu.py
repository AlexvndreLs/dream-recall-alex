"""Extract spectral, connectivity, and complexity features from preprocessed EEG.

Lit depuis derivatives/preprocessed-ica/ (sortie de preprocess_subject).
Remplace compute_psd.py, compute_cov.py, compute_cosp.py du repo Arthur.

Architecture :

Chaque enregistrement est segmenté UNE SEULE FOIS en epochs non-chevauchants
de 30s groupés par stade atomique (S1, S2, S3, S4, REM), lus depuis
_events.tsv. Les features (PSD brute, PSD oscillatoire FOOOF, exposant
aperiodic, covariance temporelle, cospectrum, entropie/complexité) sont
calculées une fois par groupe atomique et cachées sur disque.

Les états de classification (S2, SWS, REM, NREM) sont obtenus par
concaténation des tableaux atomiques cachés — sans relecture des données
brutes ni recalcul (cf CLASSIFICATION_GROUPS dans config_v3.py).

La visualisation UMAP est séparée dans visualize_umap.py, qui lit les mêmes
.npz atomiques.

Notes :

- Données en entrée : derivatives/preprocessed-ica/, 19 canaux EEG, 250Hz,
  average reference, ICA appliqué. Différent du pipeline Arthur (référence nez,
  1000Hz, pas d'ICA) -> cov/cosp non directement comparables.
- Covariances() utilise l'estimateur SCM par défaut. On n'utilise pas de shrinkage 
  statistique avancé (OAS/LW), cohérent avec le pipeline d'Arthur.
  Une régularisation numérique (diagonal loading de 1e-10) est appliquée manuellement 
  juste après pour garantir la stricte positivité de la matrice.
- Fichiers combinés .npz avec dtype=object (n_epochs variable par sujet) :
  charger avec np.load(path, allow_pickle=True).
- FOOOF (Donoghue et al. 2020, specparam) pour la séparation aperiodic/oscillatoire.
- Entropie/complexité via antropy (R. Vallat, co-auteur du dataset chapitre 1
  de la thèse, https://github.com/raphaelvallat/antropy) : permutation entropy,
  spectral entropy, Higuchi fractal dimension. Format scalaire (n_epochs, 19),
  identique à aperiodic -> classées en mode vecteur. À comparer systématiquement
  à l'exposant aperiodic seul (une mesure de complexité peut n'être qu'une
  remesure de la pente 1/f, cf Aamodt et al. 2022).
  LZC volontairement non implémentée (trop corrélée à la pente spectrale).

Usage :
    python feat_extract.py \\
        --deriv-path /path/to/derivatives/preprocessed-ica \\
        --save-path  /path/to/dream_features \\
        --n-jobs     $SLURM_CPUS_PER_TASK \\
        --overwrite  # optionnel : écrase les .npz existants

"""

import argparse
import traceback
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import mne
import antropy as ant
from specparam import SpectralGroupModel
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances, CoSpectra

from config_v3 import (
    SFREQ_PREPROC, PER_BLACKLIST_STR, JBE_SUBJECTS_STR,
    N_SAMPLES, N_EEG, CH_NAMES,
    WINDOW, OVERLAP, OVERLAP_COSP, FREQ_DICT, FOOOF_FREQ_RANGE,
    ATOMIC_STAGES, STAGE_LABEL_TO_ATOMIC,
    CLASSIFICATION_GROUPS, STATE_LIST,
    FEATURE_KEYS, SUBJECT_IDS,
)
from utils import load_atomic

SF = int(SFREQ_PREPROC)  # 250 Hz après décimation dans le prepro


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine du derivative preprocessed-ica "
                        "(ex: /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica)")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Dossier de sortie des features "
                        "(ex: /home/alouis/scratch/dream_features)")
    p.add_argument("--n-jobs", type=int, default=1, 
                   help="Parallel jobs joblib (défaut: 1 CPU)")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Écrase les .npz existants (utile après changement de params)")
    return p.parse_args()


# ─── path helpers ─────────────────────────────────────────────────────────────
#chemins vers les fichiers preprocessed (proc-clean) 
#produits par preprocess_subject_v2.py

def _vhdr(deriv_path: Path, sub_id: str) -> Path:
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_eeg.vhdr")


def _events(deriv_path: Path, sub_id: str) -> Path:
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_events.tsv")


def _choose_scorer(sub_id: str) -> str:
    if sub_id not in PER_BLACKLIST_STR:
        return "per"
    if sub_id in JBE_SUBJECTS_STR:
        return "jbe"
    raise ValueError(f"sub-{sub_id}: no valid scorer")


# ─── epoch loading (single pass per subject) ──────────────────────────────────

def load_epochs_by_atomic_stage(
    deriv_path: Path, sub_id: str
) -> dict[str, np.ndarray]:
    """
    Lit le raw preprocessé + _events.tsv une seule fois.
    Coupe des epochs non-chevauchants de 30s, groupés par stade atomique.

    Returns dict[atomic_stage] -> (n_epochs, 19, 7500) à 250Hz.
    """
    raw = mne.io.read_raw_brainvision(
        _vhdr(deriv_path, sub_id), preload=True, verbose=False
    )
    raw.pick(CH_NAMES[:N_EEG])  # selection par nom
    n_total = raw.n_times

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"

    df = pd.read_csv(_events(deriv_path, sub_id), sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = (df[df["stage"].isin(STAGE_LABEL_TO_ATOMIC)]
          .sort_values("sample")
          .reset_index(drop=True))

    epochs: dict[str, list[np.ndarray]] = {s: [] for s in ATOMIC_STAGES}

    i = 0
    while i + 29 < len(df):
        block   = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages  = block["stage"].values

        if not (np.all(samples == samples[0] + np.arange(30) * SF) and 
                np.all(stages == stages[0])):
            #on verifie si  les 30 annotations sont espacées 
            #exactement de 250 samples (1s à 250Hz), sans trou ni saut
            #et que toutes les 30 secondes appartiennent au même stade
            i += 1
            continue
        
        end = int(samples[0]) + N_SAMPLES

        # verifie que l'epoch ne depasse pas la fin du fichier on sait jamais
        if end > n_total:
            raise ValueError(
                f"sub-{sub_id}: epoch dépasse la fin du fichier "
                f"(end={end}, n_total={n_total})"
            )
        epoch = raw.get_data(start=int(samples[0]), stop=end)  # (19, 7500)
        epochs[STAGE_LABEL_TO_ATOMIC[stages[0]]].append(epoch)
        i += 30  # un pas de 30s 

    return {s: np.stack(e) for s, e in epochs.items() if e}


# ─── feature computation ──────────────────────────────────────────────────────

def compute_psd_spectrum(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_epochs, 19, 7500) -> psds (n_epochs, 19, n_freqs), freqs (n_freqs,).

    Un seul appel Welch sur le spectre complet 1-45Hz. Toutes les features
    spectrales (bandes brutes, bandes oscillatoires FOOOF) en sont dérivées
    sans second appel.
    """
    return mne.time_frequency.psd_array_welch(
        data,
        sfreq=SF,
        fmin=FOOOF_FREQ_RANGE[0],
        fmax=FOOOF_FREQ_RANGE[1],
        n_fft=WINDOW,
        n_overlap=OVERLAP,
        n_per_seg=WINDOW,
        window="hann",
        verbose=False,
    )


def band_power(
    spectrum: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float
) -> np.ndarray:
    """(n_epochs, 19, n_freqs) -> (n_epochs, 19) moyenne sur [fmin, fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax) 
    #masque booleen sur frequences ex : 1 2 3 4 true et le reste false pr prem bande
    return spectrum[..., mask].mean(axis=-1)
    #extrait la puissance moyenne sur une bande de frequences

def fit_fooof(
    psds: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit FOOOF (mode aperiodic fixe) sur chaque spectre (epoch, canal).

    specparam (ex-FOOOF) — Donoghue et al. 2020, Nature Neuroscience.
    aperiodic_mode="fixed" : pas de knee, adapté à la plage 1-45Hz
    (à réévaluer si les résultats delta SWS semblent aberrants).

    n_jobs=1 : le parallélisme est géré au niveau sujet par joblib en amont.

    Returns:
    exponent  : (n_epochs, 19)         pente aperiodic (exposant 1/f)
    flattened : (n_epochs, 19, n_freqs) psd / fit_aperiodic (ratio linéaire,
                facteur d'excès au-dessus de la pente 1/f : ~1 hors pic, >1 sur pic)
    """
    n_epochs, n_ch, n_freqs = psds.shape
    flat_psds = psds.reshape(-1, n_freqs) #specparam attend un array 2D
    #la résolution freq est SF/WINDOW = 1Hz. Donc sur 1-45Hz => nfreq = 45

    fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
    fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)
    #parallelisme est géré en amont => n_jobs =1
    # freq_range redondant (freqs déjà tronqué par psd_array_welch à FOOOF_FREQ_RANGE)
    # conservé à titre défensif pour documenter l'intention

    aperiodic = fg.get_params("aperiodic")  # specparam 2.0 : "aperiodic" (ex-"aperiodic_params" en 1.x)
    # (n_epochs*19, 2) : col0=offset , col1=exposant
    #offset = coordonées a l'origine et exponent = pente (>0)
    exponent  = aperiodic[:, 1].reshape(n_epochs, n_ch)

    offsets   = aperiodic[:, 0:1]
    exponents = aperiodic[:, 1:2]
    ap_fit_log = offsets - exponents * np.log10(freqs)[None, :]


    flat_ratio = flat_psds / (10 ** ap_fit_log)
    # résidu en RATIO LINÉAIRE (psd observée / aperiodic), pas en log :
    # band_power fait une moyenne arithmétique, valide en puissance linéaire
    # seulement -> psd_osc_* reste dans le même espace que psd_* (comparable LDA).
    # interprétation : facteur d'excès au-dessus du 1/f (~1 hors pic, >1 sur pic).

    # Aligne l'espace pour la LDA et permet la moyenne arithmétique dans band_power.
    # Interprétation : Facteur d'excès au-dessus du 1/f (~1 hors pic, >1 sur pic).
    # Ratio choisi vs soustraction linéaire : diag_foof.py (noica/REM/alpha,\
    # n=6) montre r=0.927 entre les deux -> même signal, mais soustraction\
    # en unités brutes (~1e-10) illisible sans rescaling. Ratio retenu.

    return exponent, flat_ratio.reshape(n_epochs, n_ch, n_freqs)

def compute_cov(data: np.ndarray) -> np.ndarray:
    """(n_epochs, 19, 7500) -> (n_epochs, 19, 19).

    Estimateur SCM (Sample Covariance Matrix) par défaut, cohérent avec
    le pipeline original d'Arthur. Ne pas changer en OAS/LWF sans documenter.
    """
    cov = Covariances().fit_transform(data)
    n = cov.shape[-1]
    mu = np.trace(cov, axis1=-2, axis2=-1) / n
    cov += 1e-10 * mu[:, None, None] * np.eye(n)
    return cov


def compute_cosp(
    data: np.ndarray, fmin: float, fmax: float
) -> np.ndarray:
    """(n_epochs, 19, 7500) -> (n_epochs, 19, 19) cospectrum moyen sur la bande.

    WINDOW identique à compute_psd_spectrum (250 samples, cohérence Δf).
    Overlap DIFFÉRENT volontairement : PSD à 50% (OVERLAP=125 samples),
    cospectrum à 75% (overlap=0.75, défaut pyriemann). Choix justifiés par
    la littérature Welch (1967) pour réduire la variance d'estimation.
    """
    mat = CoSpectra(
        window=WINDOW, overlap=OVERLAP_COSP, fmin=fmin, fmax=fmax, fs=SF
    ).fit_transform(data)
    # Testé hors-cluster avec pyriemann==0.11 (random data, 5 epochs x 19 ch
    # x 750 samples @ 250Hz, fmin=8/fmax=12) : CoSpectra retourne TOUJOURS du
    # 4D (n_epochs, 19, 19, n_freqs), jamais du 3D direct. Assertion stricte
    # plutôt que fallback silencieux : si une future version de pyriemann
    # change ce comportement, on veut un crash explicite, pas une matrice
    # cassée (non-SPD) qui fausserait silencieusement la classification.
    assert mat.ndim == 4, (
        f"CoSpectra a retourné du {mat.ndim}D au lieu de 4D attendu "
        f"(shape={mat.shape}) -> vérifier la version de pyriemann"
    )
    # DIAG temporaire : nombre de bins de fréquence moyennés par bande.
    # Sert à vérifier si les bandes étroites (ex. sigma 12-16Hz) ont trop
    # peu de bins à 250Hz -> motiverait le recalcul du cospectrum à 1000Hz
    # avant décimation. À retirer une fois le diagnostic fait.
    print(f"  [DIAG cosp] band={fmin}-{fmax}Hz n_freqs={mat.shape[-1]}")
    mat = mat.mean(axis=-1)
    n = mat.shape[-1]
    mu = np.trace(mat, axis1=-2, axis2=-1) / n
    mat += 1e-10 * mu[:, None, None] * np.eye(n)
    return mat


def compute_complexity(
    data: np.ndarray, spectrum: np.ndarray
) -> dict[str, np.ndarray]:
    """(n_epochs, 19, 7500) + spectre Welch -> 3 features de complexité (n_epochs, 19).

    permutation entropy et Higuchi FD opèrent sur le signal temporel brut
    (boucle epoch x canal, antropy attend du 1D). spectral entropy dérivée
    du spectre Welch déjà calculé (entropie de Shannon de la distribution de
    puissance, normalisée) -> coût quasi nul, cohérent avec la PSD.

    perm_entropy : order=3 par défaut antropy, normalize=True pour borner [0,1].
    =>quantifie l'irrégularité/imprévisibilité de l'ordre temporel des valeurs 
    en regardans si parmi les 6 permuations certaine apparaissent plus ou si toutes apparaissent pareil

    higuchi_fd   : kmax=10 par défaut antropy.
    => voir avec ariana
    => mesure à quel point le signal "remplit" le plan temps-amplitude quand on le regarde à differentes resolutions
    en gros si le signal est fractal ou irregulier a petite et grande echelle

    spec_ent : =>  est-ce que cette énergie est concentrée sur quelques fréquences,
    ou répartie uniformément sur tout le spectre ?
    welch avec OVERLAP=50% chevauchement ici => cf config_v3.py (ecart volontaire vs Arthur, perf-max)
    => Welch opère à l'intérieur d'une seule epoch, pas entre epochs
    => pas de leakage
    
    À comparer à l'exposant aperiodic (cf docstring d'en-tête).
    """
    n_epochs, n_ch, _ = data.shape

    perm_ent   = np.empty((n_epochs, n_ch))
    higuchi    = np.empty((n_epochs, n_ch))
    for ep in range(n_epochs):
        for ch in range(n_ch):
            sig = data[ep, ch]
            perm_ent[ep, ch] = ant.perm_entropy(sig, normalize=True) #m=3 mais a voir apres si plus de detail
            higuchi[ep, ch]  = ant.higuchi_fd(sig) # k= 10 => 750 echantillon => 3s 

    psd_sum = spectrum.sum(axis=-1, keepdims=True)
    psd_norm = np.divide(                                                           #(a)
        spectrum, psd_sum, out=np.zeros_like(spectrum), where=psd_sum > 0 
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_psd = np.where(psd_norm > 0, np.log2(psd_norm), 0.0)

    spec_ent = -(psd_norm * log_psd).sum(axis=-1)   #(b)
    spec_ent /= np.log2(spectrum.shape[-1])  # normalisation [0,1]

    # spectral entropy depuis le spectre Welch : 
    # Shannon normalisé (b) de la distribution de puissance (somme sur l'axe fréquences -> proba) (a)
    #on s'assure aussi non division par zero


    return {
        "perm_entropy": perm_ent,
        "higuchi_fd":   higuchi,
        "spec_entropy": spec_ent,
    }


def compute_all_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """Un groupe d'epochs -> toutes les features en un seul passage.

    Un seul appel Welch + FOOOF, toutes les features spectrales en sont dérivées.
    CoSpectra appelé une fois par bande. Complexité (perm/spectral entropy,
    Higuchi) calculée une fois, spectral entropy réutilise le spectre Welch.

    Returns dict avec clés : aperiodic, cov, psd_{band}, psd_osc_{band},
    cosp_{band}, perm_entropy, higuchi_fd, spec_entropy.
    """
    psds, freqs    = compute_psd_spectrum(data)
    exponent, flat = fit_fooof(psds, freqs)

    feats: dict[str, np.ndarray] = {
        "aperiodic": exponent,
        "cov":       compute_cov(data),
    }
    feats.update(compute_complexity(data, psds))
    for fname, (fmin, fmax) in FREQ_DICT.items():
        feats[f"psd_{fname}"]     = band_power(psds,  freqs, fmin, fmax)
        feats[f"psd_osc_{fname}"] = band_power(flat,  freqs, fmin, fmax)
        feats[f"cosp_{fname}"]    = compute_cosp(data, fmin, fmax)
    return feats


# ─── per-subject pipeline ─────────────────────────────────────────────────────

def process_subject(
    deriv_path: Path, save_path: Path, sub_id: str, overwrite: bool = False
) -> None:
    if not _vhdr(deriv_path, sub_id).exists():
        print(f"sub-{sub_id}: derivative not found, skipping")
        return
    #Verifie que le fichier preprocessé existe sur disque

    try:
        atomic_epochs = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        print(f"sub-{sub_id}: ERROR loading\n{traceback.format_exc()}")
        return
    #charge le raw et decoupe en epochs

    for stage, data in atomic_epochs.items():
        print(f"  sub-{sub_id} {stage}: {data.shape[0]} epochs")

        # skip si tous les .npz de ce sujet/stade existent -> reprise apres crash cluster
        if not overwrite and all(
            (save_path / k / f"{k}_s{sub_id}_{stage}.npz").exists()
            for k in FEATURE_KEYS
        ):
            print(f"  sub-{sub_id} {stage}: already cached, skipping")
            continue
        
        try:
            feats = compute_all_features(data)
        except Exception:
            print(f"sub-{sub_id} {stage}: ERROR features\n{traceback.format_exc()}")
            continue
        #calcul des features

        for key, arr in feats.items():
            out = save_path / key / f"{key}_s{sub_id}_{stage}.npz"
            # double check par feature : protege contre un crash entre deux saves
            if not out.exists() or overwrite:
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, data=arr)
                #sauvegarde l'array compressé
                
    print(f"sub-{sub_id}: done")

# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args       = parse_args()
    deriv_path = args.deriv_path
    save_path  = args.save_path
    n_jobs     = args.n_jobs
    overwrite  = args.overwrite

    t0 = time()

    print("=== extraction features par sujet (stades atomiques) ===")
    Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(deriv_path, save_path, sub_id, overwrite)
        for sub_id in SUBJECT_IDS
    )

    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")
    print("Lancer visualize_umap.py --save-path <save_path> pour le UMAP.")
