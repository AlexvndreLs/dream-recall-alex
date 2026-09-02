"""t-test HR vs LR au niveau SUJET (RFX) sur n'importe quelle feature vectorielle,
avec correction par statistique du maximum sur les 19 electrodes.

Pourquoi ce script existe
-------------------------
recompute_ttest_fig3.py a BANDS = list(FREQ_DICT) code en dur et ne charge que
psd_{band}. Il ne sait donc pas traiter aperiodic, aperiodic_offset, psd_osc_*,
psd_sub_* ni psd_logsub_*. Ce script generalise a une key arbitraire, et ne
garde que le schema RFX, seul retenu pour le rapport.

Ce qu'il fait
-------------
Pour chaque couple (key, state) :
  1. charge les epochs de chaque sujet (concatenation des stades atomiques de
     CLASSIFICATION_GROUPS[state], comme classify.py via load_subject),
  2. moyenne sur les epochs -> une valeur par sujet et par electrode,
  3. t de Welch HR vs LR par electrode,
  4. p corrigee par max-stat sur les 19 electrodes, permutation des labels
     sujet.

Convention de signe : cond1 = HR, cond2 = LR. Un t positif signifie valeur plus
haute chez les hauts rappeleurs. Pour une figure au format Tholke et al. 2025,
rouge = t positif, bleu = t negatif.

Portee de la correction
-----------------------
Max-stat sur les 19 electrodes d'UNE key dans UN etat, jamais de pool entre
keys ni entre bandes. C'est la portee de Tholke et al., verifiee dans leur code
(statistics/Statistics.ipynb : un appel mne.stats.permutation_t_test par couple
stade x feature, la correction porte sur la dimension des canaux). C'est aussi
la portee du mode 'arthur' de compute_maxstat_correction.py, donc les colonnes
t-values et LDA de la figure seront corrigees au meme barème.

La fonction d'inference n'est pas reecrite : ttest_maxstat est importee telle
quelle de recompute_ttest_fig3.py. Meme code, memes conventions, un seul
endroit a auditer.

Ecart avec Tholke assume : leur contraste est apparie (deux nuits du meme
sujet, t-test sur les differences), le notre oppose deux groupes de sujets
differents, d'ou ttest_ind avec equal_var=False. Structurellement moins
puissant, les deux figures ne se comparent pas terme a terme.

Usage
-----
    # exposant + offset, branche overlap
    python ttest_vector_rfx.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_overlap_ttest \
        --keys aperiodic aperiodic_offset \
        --n-perm 9999

    # definition brute et definition ratio, meme branche
    python ttest_vector_rfx.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_overlap_ttest \
        --keys psd_delta psd_theta psd_alpha psd_sigma psd_beta \
               psd_osc_delta psd_osc_theta psd_osc_alpha psd_osc_sigma psd_osc_beta

    # definition sub et definition logsub : branches separees
    python ttest_vector_rfx.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_sub \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_sub_ttest \
        --keys psd_sub_delta psd_sub_theta psd_sub_alpha psd_sub_sigma psd_sub_beta

n-jobs : laisser 1. Chaque permutation est un ttest_ind sur (36, 19), l'overhead
joblib depasse largement le calcul. A n-jobs 1, un couple key x state prend
moins d'une seconde a 9999 permutations.

Sorties
-------
    {out-dir}/{key}_{state}_ttest_rfx.npz
        tvals             (19,)  t de Welch, signe HR moins LR
        pvals_corrected   (19,)  max-stat sur les 19 electrodes
        pvals_uncorrected (19,)  t-test parametrique, pour reference seulement
        ch_names          (19,)
        n_cond1, n_cond2, n_perm, subjects_used
    {out-dir}/ttest_rfx_summary.csv
"""

import argparse
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from config_v3 import (
    CH_NAMES,
    N_EEG,
    SUBJECT_LIST_ORDERED,
    SUBJECT_LABELS,
    CLASSIFICATION_GROUPS,
    STATE_LIST,
)
from utils import load_atomic
# Reutilise l'inference de recompute_ttest_fig3 plutot que de la reecrire.
# L'import est sans effet de bord : ce module ne contient au niveau global
# qu'un BANDS = list(FREQ_DICT) et le garde __main__.
from recompute_ttest_fig3 import ttest_maxstat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                   help="Racine de la branche de features (contient {key}/ )")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Dossier de sortie. Jamais le meme que --save-path.")
    p.add_argument("--keys", nargs="+", required=True,
                   help="Keys vectorielles a traiter (ex: aperiodic psd_osc_sigma)")
    p.add_argument("--states", nargs="+", default=STATE_LIST)
    p.add_argument("--n-perm", type=int, default=9999,
                   help="Permutations des labels sujet. 9999 -> p minimale 1/9999.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Laisser 1 : les permutations sont trop courtes pour "
                        "amortir l'overhead joblib.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-subjects", type=str, default="",
                   help="Numeros de sujets a exclure, separes par des virgules. "
                        "Vide par defaut, pour rester sur la meme cohorte que "
                        "classify.py (n_subjects = 36).")
    p.add_argument("--one-tailed", action="store_true", default=False,
                   help="Par defaut bilateral, comme Tholke et al. et comme le "
                        "texte de la these. Ne pas activer sans raison.")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def load_subject_means(save_path: Path, key: str, state: str,
                       drop_ids: set) -> tuple[np.ndarray, np.ndarray, list]:
    """-> means (n_sujets, 19), labels (n_sujets,), liste des sujets retenus.

    Concatenation des stades atomiques puis moyenne sur les epochs : une valeur
    par sujet et par electrode. C'est le passage au niveau sujet qui fait le
    RFX. Un sujet sans donnee dans cet etat est simplement absent, comme dans
    load_all de classify.py.
    """
    means, labels, used = [], [], []
    stages = CLASSIFICATION_GROUPS[state]
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        if sub_id in drop_ids:
            continue
        parts = [a for s in stages
                 if (a := load_atomic(save_path, key, sub_id, s)) is not None]
        if not parts:
            continue
        arr = np.concatenate(parts, axis=0)          # (n_epochs, 19)
        means.append(arr.mean(axis=0))               # (19,)
        labels.append(label)
        used.append(sub_id)
    if not means:
        return np.empty((0, N_EEG)), np.empty(0, dtype=int), []
    return np.vstack(means), np.array(labels), used


def save_atomic_npz(out: Path, **arrays) -> None:
    """Ecriture atomique : temporaire puis os.replace. Ne supprime que son
    propre temporaire en cas d'echec."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    try:
        # np.savez_compressed ajoute ".npz" a un chemin qui n'en a pas :
        # passer un descripteur ouvert evite que le fichier ecrit ne porte
        # pas le nom attendu par os.replace.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


if __name__ == "__main__":
    args = parse_args()
    if args.out_dir.resolve() == args.save_path.resolve():
        raise SystemExit("--out-dir doit differer de --save-path.")

    drop_ids = {s.strip().zfill(2) for s in args.drop_subjects.split(",") if s.strip()}
    two_tailed = not args.one_tailed
    ch_names = np.array(CH_NAMES[:N_EEG])

    print("=== t-test RFX, max-stat sur electrodes ===")
    print(f"branche  : {args.save_path}")
    print(f"sortie   : {args.out_dir}")
    print(f"keys     : {' '.join(args.keys)}")
    print(f"etats    : {' '.join(args.states)}")
    print(f"n_perm   : {args.n_perm}   bilateral : {two_tailed}")
    print(f"exclus   : {sorted(drop_ids) if drop_ids else 'aucun'}")
    print()

    rows = []
    t0 = time()

    for key in args.keys:
        for state in args.states:
            out = args.out_dir / f"{key}_{state}_ttest_rfx.npz"
            if out.exists() and not args.overwrite:
                print(f"{key} x {state} : deja calcule, skip")
                continue

            means, labels, used = load_subject_means(args.save_path, key, state, drop_ids)
            if len(means) < 4:
                print(f"{key} x {state} : cohorte insuffisante (n={len(means)}), skip")
                continue

            cond1 = means[labels == 1]   # HR
            cond2 = means[labels == 0]   # LR
            if len(cond1) < 2 or len(cond2) < 2:
                print(f"{key} x {state} : un groupe trop petit "
                      f"(HR={len(cond1)}, LR={len(cond2)}), skip")
                continue

            tvals, pvals_corr = ttest_maxstat(
                cond1, cond2,
                n_perm=args.n_perm,
                two_tailed=two_tailed,
                seed=args.seed,
                n_jobs=args.n_jobs,
                arthur_pval_bug=False,
            )
            p_uncorr = ttest_ind(cond1, cond2, equal_var=False)[1]

            save_atomic_npz(
                out,
                tvals=tvals,
                pvals_corrected=pvals_corr,
                pvals_uncorrected=p_uncorr,
                ch_names=ch_names,
                n_cond1=len(cond1),
                n_cond2=len(cond2),
                n_perm=args.n_perm,
                subjects_used=np.array(used),
            )

            i = int(np.abs(tvals).argmax())
            n5 = int((pvals_corr < 0.05).sum())
            n1 = int((pvals_corr < 0.01).sum())
            print(f"{key:20s} {state:4s} HR={len(cond1)} LR={len(cond2)}  "
                  f"|t|max={abs(tvals[i]):5.2f} @{str(ch_names[i]):4s} "
                  f"t={tvals[i]:+6.2f}  p_corr={pvals_corr[i]:.4f}  "
                  f"n<.05={n5} n<.01={n1}")

            for e, ch in enumerate(ch_names):
                rows.append(dict(key=key, state=state, electrode=ch,
                                 tval=float(tvals[e]),
                                 pval_corrected=float(pvals_corr[e]),
                                 pval_uncorrected=float(p_uncorr[e]),
                                 n_hr=len(cond1), n_lr=len(cond2),
                                 n_perm=args.n_perm))

    if rows:
        csv = args.out_dir / "ttest_rfx_summary.csv"
        df = pd.DataFrame(rows)
        if csv.exists():
            old = pd.read_csv(csv)
            df = (pd.concat([old, df])
                    .drop_duplicates(subset=["key", "state", "electrode"], keep="last"))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"\nCSV : {csv}  ({len(df)} lignes)")

    m, s = divmod(int(time() - t0), 60)
    print(f"total: {m}m{s:02d}s")
