"""Extraction standalone de la complexite de Lempel-Ziv (LZ76).

Complete les trois mesures de complexite deja extraites (perm_entropy,
higuchi_fd, spec_entropy) par feat_extract_umap_fooof_v4.py, qui laissait
volontairement la LZC de cote (cf sa docstring d'en-tete : "LZC volontairement
non implementee (trop correlee a la pente spectrale)").

Pourquoi un script separe plutot qu'un patch de feat_extract_umap_fooof_v4.py
-----------------------------------------------------------------------------
process_subject() saute un couple (sujet, stade) si TOUS les .npz de
FEATURE_KEYS existent deja. Ajouter "lzc" a FEATURE_KEYS casserait ce test
pour les 190 couples deja caches et relancerait Welch + FOOOF + 5 CoSpectra
sur toute la cohorte, pour une seule colonne supplementaire. Ce script
reutilise donc load_epochs_by_atomic_stage() par import direct, ce qui
garantit une segmentation strictement identique (memes epoques, memes bornes,
meme choix de scorer), et n'ecrit que le dossier de la cle demandee.

Format de sortie
----------------
{save_path}/{key_name}/{key_name}_s{XX}_{stage}.npz
    data   : (n_epochs, 19) float64
    params : chaine JSON decrivant la definition exacte de la mesure

Format scalaire par electrode, identique a aperiodic / higuchi_fd /
perm_entropy / spec_entropy, donc directement classifiable par
classify.py --key lzc --skip-check (route vectorielle, LDA par electrode).

Definition et bande passante
----------------------------
La branche noica n'a AUCUN passe-bas : preprocess_subject_v3.py applique un
notch 50/100 Hz et un passe-haut 0.1 Hz, rien d'autre. A 1000 Hz le signal
contient donc tout jusqu'a Nyquist, EMG compris. Or la LZC binarisee mesure
au premier ordre la structure de la suite des croisements de la mediane, et
le taux de croisement est fixe par le bord haut du spectre. Sans passe-bas,
la mesure devient un proxy de puissance haute frequence.

Ce que fait la litterature, verifie dans les sources :
  Tholke et al. 2025 (Commun Biol), l'etude qui a motive cette section :
      enregistrement 256 Hz, passe-bande 0.5-32 Hz applique aux epoques
      avant extraction, median split sur le signal, normalisation
      log_b(n)/n, identique a antropy normalize=True.
  Aamodt et al. 2022 (Front Hum Neurosci), LZC et experience onirique :
      sous-echantillonnage a 250 Hz, passe-haut 0.75 Hz, epoques 30 s,
      binarisation de l'amplitude de Hilbert.
  Hohn et al. 2024 (eNeuro), pente spectrale et LZC :
      250 Hz, binarisation de l'amplitude de Hilbert autour de sa mediane.
Aucune de ces etudes ne calcule la LZC sur un signal non filtre.

Defaut retenu ici : passe-bande 1-45 Hz, sans decimation, on reste a 1000 Hz.
La borne haute 45 Hz est FOOOF_FREQ_RANGE, donc la meme bande que spec_entropy
et surtout que l'exposant aperiodique contre lequel le protocole de redondance
confronte la mesure. Pour la version strictement Tholke, passer
--highpass 0.5 --lowpass 32.

Ce qui compte est le filtrage, pas la frequence d'echantillonnage. Mesure sur
un 1/f borne 1-45 Hz auquel on ajoute du bruit 100-400 Hz valant 5 pourcent de
la variance totale :

    lzc           0.1403 -> 0.3574, et 0.1343 apres passe-bas 45 Hz
    perm_entropy  0.5279 -> 0.9863, et 0.5175 apres passe-bas 45 Hz
    higuchi_fd    1.0507 -> 1.7125, et 1.0409 apres passe-bas 45 Hz

Cinq pourcent de variance hors bande deplacent la LZC de 155 pourcent, et le
passe-bas la ramene a sa valeur propre. La decimation, elle, ne change rien :
n etant identique pour toutes les epoques, la normalisation par n/log2(n) est
une constante multiplicative, et la LDA a une dimension par electrode est
invariante par changement d'echelle. --decimate 4 ne sert donc qu'a diviser le
temps de calcul par environ 20, le comptage LZ76 etant quadratique en
longueur. C'est un reglage de confort.

Limite a signaler dans le rapport : higuchi_fd et perm_entropy sont, elles,
calculees sur le signal large bande non filtre a 1000 Hz. Aligner la LZC sur
la litterature cree donc une asymetrie dans le tableau. L'inverse produirait
un chiffre non interpretable. Le bras de controle --no-filter permet de
mesurer l'ecart plutot que de l'affirmer.

Usage
-----
    # recommande : bande limitee 1-45 Hz, 1000 Hz conserve
    python feat_extract_lzc.py \
        --deriv-path /scratch/alouis/dream_bids/derivatives_1000hz/preprocessed-noica \
        --save-path  /scratch/alouis/dream_features_noica_1000hz \
        --n-jobs     $SLURM_CPUS_PER_TASK

    # bras de controle : signal brut 1000 Hz, meme entree que higuchi/perm_entropy
    python feat_extract_lzc.py ... --no-filter --key-name lzc_raw

    # variante Tholke stricte
    python feat_extract_lzc.py ... --highpass 0.5 --lowpass 32 --key-name lzc_tholke

    # variante Aamodt / Hohn : enveloppe de Hilbert
    python feat_extract_lzc.py ... --hilbert --key-name lzc_env

    # test incremental sur un sujet avant de lancer le job complet
    python feat_extract_lzc.py ... --subjects 01 --n-jobs 1
"""

import argparse
import json
import traceback
from pathlib import Path
from time import time

import numpy as np
import antropy as ant
from joblib import Parallel, delayed
from scipy.signal import butter, decimate, filtfilt, hilbert

from config_v3 import SFREQ_PREPROC, SUBJECT_IDS
from feat_extract_umap_fooof_v4 import load_epochs_by_atomic_stage, _vhdr

SF = int(SFREQ_PREPROC)


def normalize_subject_ids(raw_ids, parser) -> list:
    """Force le zero-padding sur deux chiffres des IDs passes en CLI.

    Indispensable, et pour deux raisons distinctes.

    1. utils.load_atomic() applique str(sub_id).zfill(2) a la lecture. Ecrire
       lzc_s1_S2.npz produirait un fichier que classify.py ne trouvera jamais,
       sans erreur : c'est le bug de zero-padding qui avait invalide les
       resultats matriciels (cov_s1_S2.npz au lieu de cov_s01_S2.npz).

    2. _choose_scorer() compare sub_id a PER_BLACKLIST_STR et JBE_SUBJECTS_STR,
       donc par chaine. Avec "1" au lieu de "01" la blacklist ne matche pas, le
       scorer per est retenu silencieusement, et les epoques sont decoupees sur
       le mauvais scoring. Meme classe de bug que le 10 in {"10"} historique.

    Les deux echouent SANS lever d'exception, d'ou la normalisation en amont
    plutot qu'une simple verification.
    """
    out = []
    for s in raw_ids:
        s = str(s).strip()
        if not s.isdigit():
            parser.error(f"--subjects : '{s}' n'est pas un identifiant numerique.")
        padded = s.zfill(2)
        if padded not in SUBJECT_IDS:
            parser.error(f"--subjects : sub-{padded} absent de SUBJECT_IDS "
                         f"(config_v3.py).")
        out.append(padded)
    return out


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine du derivative preprocessed-*, meme valeur que "
                        "pour feat_extract_umap_fooof_v4.py.")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine des features, ex /scratch/alouis/dream_features_noica_1000hz")
    p.add_argument("--key-name", type=str, default="lzc",
                   help="Nom du dossier et prefixe de fichier.")

    p.add_argument("--highpass", type=float, default=1.0,
                   help="Passe-haut Butterworth ordre 4, en Hz.")
    p.add_argument("--lowpass", type=float, default=45.0,
                   help="Passe-bas Butterworth ordre 4, en Hz.")
    p.add_argument("--decimate", type=int, default=1,
                   help="Facteur de decimation, FIR anti-alias a phase nulle, "
                        "applique apres le filtrage. 1 = aucune, on reste a "
                        "1000 Hz. Decimer un signal deja borne a 45 Hz est sans "
                        "perte (Nyquist) et ne change pas les accuracies, cela "
                        "ne fait qu'accelerer le comptage LZ76. Option de "
                        "confort, pas de methode.")
    p.add_argument("--no-filter", action="store_true", default=False,
                   help="Desactive filtrage ET decimation : signal brut 1000 Hz. "
                        "Bras de controle, environ 20x plus lent.")

    p.add_argument("--hilbert", action="store_true", default=False,
                   help="Binarise l'amplitude instantanee de Hilbert plutot que "
                        "le signal, convention Aamodt 2022 et Hohn 2024. Par "
                        "defaut on binarise le signal, convention Tholke 2025.")
    p.add_argument("--binarize", choices=["median", "mean"], default="median",
                   help="Seuil de binarisation, par epoque et par electrode.")

    p.add_argument("--subjects", nargs="+", default=None,
                   help="Sous-ensemble d'IDs BIDS, ex : 01 02. Defaut : tous.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", default=False)

    args = p.parse_args()
    if args.no_filter:
        args.highpass, args.lowpass, args.decimate = None, None, 1
    if args.subjects:
        args.subjects = normalize_subject_ids(args.subjects, p)
    if args.decimate < 1:
        p.error("--decimate doit valoir au moins 1.")
    if args.lowpass is not None and args.decimate > 1:
        nyq_out = SF / args.decimate / 2.0
        if args.lowpass >= nyq_out:
            p.error(f"--lowpass {args.lowpass} >= Nyquist apres decimation "
                    f"({nyq_out} Hz). Baisser --lowpass ou --decimate.")
    return args


def build_params(args: argparse.Namespace) -> str:
    """Signature JSON de la definition exacte de la mesure.

    Ecrite dans chaque .npz et comparee aux .npz deja presents dans le dossier
    de la cle : deux definitions differentes ne peuvent pas cohabiter sous la
    meme cle, meme par accident.
    """
    return json.dumps(dict(
        measure="lempel_ziv_1976",
        implementation="antropy.lziv_complexity",
        normalize="n / log2(n)",
        binarize=args.binarize,
        on="hilbert_amplitude" if args.hilbert else "signal",
        highpass_hz=args.highpass,
        lowpass_hz=args.lowpass,
        decimate=args.decimate,
        sfreq_in=SF,
        sfreq_effective=SF / args.decimate,
    ), sort_keys=True)


def check_params_consistency(out_dir: Path, params_json: str) -> None:
    """Refuse d'ecrire dans un dossier peuple par une autre definition."""
    if not out_dir.exists():
        return
    for f in sorted(out_dir.glob("*.npz")):
        with np.load(f, allow_pickle=True) as d:
            existing = str(d["params"]) if "params" in d else None
        if existing is None:
            raise SystemExit(
                f"{f} n'a pas de champ 'params' : il vient d'une version "
                f"anterieure de ce script, definition inconnue. Deplacer le "
                f"dossier {out_dir} avant de relancer."
            )
        if existing != params_json:
            raise SystemExit(
                f"Definition incompatible dans {out_dir}.\n"
                f"  deja present : {existing}\n"
                f"  demande      : {params_json}\n"
                f"Utiliser un --key-name distinct, ou deplacer le dossier."
            )
        return  # un seul fichier suffit a trancher


# --- coeur ------------------------------------------------------------------

def design_filter(highpass: float | None, lowpass: float | None):
    """Coefficients Butterworth ordre 4, calcules une seule fois par sujet."""
    nyq = SF / 2.0
    if highpass is not None and lowpass is not None:
        return butter(4, [highpass / nyq, lowpass / nyq], btype="band")
    if lowpass is not None:
        return butter(4, lowpass / nyq, btype="low")
    if highpass is not None:
        return butter(4, highpass / nyq, btype="high")
    return None


def preprocess_epoch(ep: np.ndarray, ba, q: int) -> np.ndarray:
    """(19, n) -> (19, n') filtre puis decime, UNE epoque a la fois.

    Traiter epoque par epoque plutot que le stade entier est deliberé. filtfilt
    alloue un tableau de la taille de son entree, plus son padding interne :
    sur le S2 d'une nuit longue cela represente plusieurs Go alloues pour rien,
    alors que la LZC est de toute facon comptee epoque par epoque. Ici le
    surcout memoire est d'une seule epoque, soit environ 4.6 Mo.

    filtfilt traite chaque serie 1D independamment le long de axis=-1, et le
    padding ne depend que de l'ordre du filtre : le resultat est donc identique
    BIT A BIT a un filtrage du stade complet. Verifie numeriquement.

    filtfilt et decimate(zero_phase=True) sont a phase nulle : pas de decalage
    temporel qui biaiserait le comptage LZ.
    """
    out = ep
    if ba is not None:
        out = filtfilt(ba[0], ba[1], out, axis=-1)
    if q > 1:
        out = decimate(out, q, axis=-1, ftype="fir", zero_phase=True)
    return np.ascontiguousarray(out)


def binarize(sig: np.ndarray, mode: str) -> np.ndarray:
    """Serie 1D -> chaine binaire uint32, seuil mediane ou moyenne.

    Convention >= seuil, celle de la doc antropy. Le sens de l'inegalite est
    sans effet : echanger 0 et 1 dans toute la chaine ne change pas le nombre
    de motifs distincts, donc pas la LZC.
    """
    thr = np.median(sig) if mode == "median" else np.mean(sig)
    return (sig >= thr).astype(np.uint32)


def compute_lzc(data: np.ndarray, mode: str, use_hilbert: bool,
                ba, q: int) -> np.ndarray:
    """(n_epochs, 19, n) -> (n_epochs, 19), LZ76 normalise par n / log2(n).

    antropy.lziv_complexity(normalize=True) divise par n / log_b(n) avec b le
    nombre de symboles distincts, soit 2 ici. C'est exactement la
    normalisation annoncee dans la section 3 du rapport, et celle de
    Tholke et al. 2025.

    Le noyau _lz_complexity est compile par numba : la boucle Python
    epoque x electrode ne pese rien face au comptage lui-meme. Verifier que
    numba est bien installe dans l'environnement, sans lui antropy retombe
    sur du Python pur et le calcul devient inutilisable.
    """
    n_epochs, n_ch = data.shape[0], data.shape[1]
    out = np.empty((n_epochs, n_ch))
    for ep in range(n_epochs):
        proc = preprocess_epoch(data[ep], ba, q)
        if use_hilbert:
            proc = np.abs(hilbert(proc, axis=-1))
        for ch in range(n_ch):
            out[ep, ch] = ant.lziv_complexity(
                binarize(proc[ch], mode), normalize=True
            )
    return out


# --- pipeline par sujet -----------------------------------------------------

def process_subject(
    deriv_path: Path, save_path: Path, sub_id: str, key_name: str,
    mode: str, use_hilbert: bool, highpass: float | None,
    lowpass: float | None, q: int, overwrite: bool, params_json: str,
) -> None:
    if not _vhdr(deriv_path, sub_id).exists():
        print(f"sub-{sub_id}: derivative absent, skip", flush=True)
        return

    out_dir = save_path / key_name
    ba = design_filter(highpass, lowpass)
    try:
        atomic_epochs = load_epochs_by_atomic_stage(deriv_path, sub_id)
    except Exception:
        print(f"sub-{sub_id}: ERREUR chargement\n{traceback.format_exc()}", flush=True)
        return

    # pop plutot qu'iteration : libere chaque stade des qu'il est traite.
    # Une nuit complete a 1000 Hz pese plusieurs Go par sujet.
    for stage in sorted(atomic_epochs):
        out = out_dir / f"{key_name}_s{sub_id}_{stage}.npz"
        if out.exists() and not overwrite:
            atomic_epochs.pop(stage)
            print(f"  sub-{sub_id} {stage}: deja en cache, skip", flush=True)
            continue

        data = atomic_epochs.pop(stage)
        t0 = time()
        try:
            arr = compute_lzc(data, mode, use_hilbert, ba, q)
        except Exception:
            print(f"sub-{sub_id} {stage}: ERREUR lzc\n{traceback.format_exc()}", flush=True)
            del data
            continue
        del data

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, data=arr, params=np.array(params_json))
        print(f"  sub-{sub_id} {stage}: {arr.shape[0]} epochs, {time() - t0:.0f}s, "
              f"mean={arr.mean():.4f} sd={arr.std():.4f}", flush=True)

    print(f"sub-{sub_id}: done", flush=True)


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    subjects = args.subjects or SUBJECT_IDS
    params_json = build_params(args)
    check_params_consistency(args.save_path / args.key_name, params_json)

    print("=== extraction LZC (stades atomiques) ===")
    print(f"cle        : {args.key_name}")
    print(f"parametres : {params_json}")
    print(f"sujets     : {len(subjects)}  |  n_jobs = {args.n_jobs}")
    if args.lowpass is None:
        print("ATTENTION : aucun passe-bas. La branche noica monte jusqu'a "
              "Nyquist, EMG inclus, et la LZC devient un proxy de puissance "
              "haute frequence. A n'utiliser que comme bras de controle.")

    t0 = time()
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(
            args.deriv_path, args.save_path, sub_id, args.key_name,
            args.binarize, args.hilbert, args.highpass, args.lowpass,
            args.decimate, args.overwrite, params_json,
        )
        for sub_id in subjects
    )
    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")
    print(f"Suite : classify.py --key {args.key_name} --skip-check")