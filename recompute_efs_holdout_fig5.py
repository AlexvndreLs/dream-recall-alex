"""Recompute Fig. 5 (these Arthur chap.1) - SCRIPT 1/2 : EFS par electrode + holdout.

REPLIQUE EXACTE d'Arthur (EFS_fixed_elec.py du repo arthurdehgan/sleep), SANS
subsampling (version corrigee de la these : "data without subsampling, resulting in
minimal changes to decoding accuracy scores").

Ce script produit, POUR CHAQUE ELECTRODE :
  - test_scores : accuracy holdout (sur 2 sujets left-out) par split leave-2-out.
  - best_freqs  : bandes selectionnees par l'EFS niche a chaque split.
  - score       : moyenne des test_scores (accuracy holdout NON BIAISEE).
Les selection rates par bande et l'agregation par ROI (pie charts Fig.5 gauche) et le
barplot d'accuracy (Fig.5 droite) sont produits au stade PLOT, en agregeant ces
sorties par ROI. Les p-values p<0.001 sont ajoutees par le SCRIPT 2/2
(recompute_efs_perms_fig5.py), qui recharge best_freqs et permute (comme Arthur qui
separe EFS_fixed_elec.py et permutations_EFS_fixed_elec.py).

Mecanique (fidele EFS_fixed_elec.py, lignes 29-123)
---------------------------------------------------
Pour une electrode, on charge les 5 bandes PSD -> par sujet un array (n_epochs, 5)
(5 = features candidates : delta, theta, alpha, sigma, beta a CETTE electrode).
Split externe = StratifiedLeave2GroupsOut (1 HR + 1 LR en test, 18x18=324 splits).
Pour chaque split :
  1. x_train = epochs des 34 sujets d'entrainement (concatenees), y au niveau epoch.
  2. EFS niche (mlxtend ExhaustiveFeatureSelector, LDA, max_features=5, CV interne =
     StratifiedLeave2GroupsOut nichee sur les 34 sujets) -> best_idx (bandes retenues).
  3. Refit LDA sur x_train[:, best_idx], score sur les 2 sujets held-out -> test_score
     NON BIAISE (les 2 sujets n'ont jamais servi a la selection).
On garde TOUTES les epochs (pas de subsampling).

Fidelite / choix
----------------
- mlxtend EFS comme Arthur (fidelite++, modernisation plus tard). min_features=1,
  max_features=5 -> les 31 sous-ensembles non vides des 5 bandes.
- StratifiedLeave2GroupsOut importe de classify.py (identique a la classe d'Arthur,
  absente de son utils public : 1 HR + 1 LR par test).
- Features = tes psd_{band} existantes, slicees par electrode. AUCUN recompute.
- Permutation niveau epoch : geree par le script 2/2 (comme Arthur).

Entrees : {save_path}/psd_{band}/psd_{band}_s{XX}_{state}.npz (cle "data", (n_epochs,19)).
Sorties : {out_dir}/efs_holdout_{state}.npz
          Par electrode : test_scores (324,), best_freqs (liste 324 x sous-ensemble),
          train_scores (324,), score (float). Empile les 19 electrodes.

======================= ECARTS AVEC LE CODE D'ARTHUR (documentes) ==============
Comparaison ligne par ligne avec EFS_fixed_elec.py d'Arthur. Ecarts :

E1. PARALLELISATION (ecart de perf, resultat identique). Arthur : EFS(..., n_jobs=-1)
    -> mlxtend parallelise en interne (threads). Nous : mlxtend n_jobs=1 + Parallel(
    prefer="processes") au niveau des 324 splits externes. Raison : sur AMD EPYC, le
    nested parallelism threads + GIL + contention BLAS ecroule les perfs ; processus +
    BLAS mono-thread = ~5-6x plus rapide (cf optimisation classify_matrix). Resultat
    NUMERIQUE identique (memes splits, meme EFS deterministe), seule la vitesse change.

E2. min_features (non-ecart). Arthur : EFS(max_features=5) sans min_features (defaut
    mlxtend = 1). Nous : min_features=1, max_features=5 explicite. IDENTIQUE en pratique
    (teste les 31 sous-ensembles non vides des 5 bandes).

E3. SANS SUBSAMPLING (aligne sur la version CORRIGEE d'Arthur). Le texte de la these
    (p52) mentionne un subsampling a n_trials=61 epochs/sujet dans la version
    INITIALE. La version CORRIGEE de la these le retire explicitement ("data without
    subsampling, resulting in minimal changes to decoding accuracy scores and slight
    adjustments to significance levels", p58). Nous suivons la version corrigee : on
    garde TOUTES les epochs. (L'ancien classify_multifeature.py du repo, lui,
    subsamplait -> ce script s'en distingue volontairement.)

E4. SUJET 10 GARDE (fidele, mais incoherence d'Arthur). Arthur GARDE le sujet 10 dans
    l'EFS (lil_labels = [0]*18+[1]*18, 36 sujets), alors qu'il l'EXCLUT dans le ttest
    (Fig.3). Incoherence d'Arthur, reproduite ici : on garde 36 sujets pour la Fig.5.
    NB : sujet 10 = outlier FC2 (delta 23.7x mediane), donc son maintien peut gonfler
    l'accuracy fronto-centrale. A documenter dans la note.

E5. EFS PAR ELECTRODE, ROI a la VISU (fidele). Arthur fait l'EFS par electrode puis
    agrege par ROI au stade visu (visu_piecharts_fselect.py). Nous idem (agregation
    dans aggregate_roi_fig5.py). A ne pas confondre avec classify_multifeature.py qui
    faisait l'EFS PAR ROI (moyenne des electrodes) : ce script fait bien PAR ELECTRODE.

E6. MONTAGE 12 vs 19 ELECTRODES (ecart de donnees, affecte les accuracies). Arthur a
    12 electrodes (Fz, Cz, Pz, Fp1, F3, FC1, C3, T3, CP1, P3, M1, O1 : montage
    demi-gauche gauche), nous 19 (deux hemispheres). L'EFS tourne sur nos 19, donc on
    a des electrodes qu'Arthur n'avait pas (Fp2, F4, C4, T4, P4, O2, FC2, CP2). Ca ne
    biaise pas l'EFS par electrode (chaque electrode est traitee independamment), mais
    ca change l'AGREGATION par ROI (nos ROI ont plus d'electrodes). M1 (mastoide)
    absent de nos donnees EEG (misc1/2/3 non identifies, pas de features). cf variante
    aggregate_roi_fig5_arthur11.py (agregation sur les 11 electrodes d'Arthur moins M1).

POINTS VERIFIES IDENTIQUES (non-ecarts, documentes pour transparence) :
 - Groupes de la CV nichee : Arthur utilise create_groups qui RENUMEROTE les sujets du
   train 0..33 ; nous gardons les indices originaux (0..35 en sautant les 2 sujets de
   test). Fonctionnellement IDENTIQUE : le splitter leave-2-groups-out ne depend que de
   l'unicite des groupes par sujet, pas de leur valeur.
 - Reconstruction best_idx : Arthur fait .strip().capitalize() sur les noms de bandes
   (ses .mat ont des espaces/casse variable) ; nos noms sont propres ('sigma' etc.),
   donc pas besoin. Dependance implicite : les noms de bandes doivent matcher entre
   script 1 (EFS) et script 2 (perms) -> garanti car les deux font BANDS=list(FREQ_DICT).
 - Bandes alpha(8-13)/sigma(11-16) se CHEVAUCHENT (11-13 Hz). Decoupage d'Arthur,
   fidele, mais les features alpha et sigma ne sont pas independantes sur 11-13 Hz.
 - EFS.fit(x, y, groups) : Arthur passe groups en positionnel, nous en keyword. Idem.
================================================================================
================================================================================

Usage
-----
    python recompute_efs_holdout_fig5.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_corrected/fig5_recompute \
        --state     S2 \
        --n-jobs    $SLURM_CPUS_PER_TASK

ATTENTION cout : 19 electrodes x 324 splits x EFS(31 sous-ensembles x CV nichee 324).
C'est le gros calcul. Valider sur 1 electrode (--only-elec Fz) avant le batch complet.

Author: recompute pour Alex (replique Arthur chap.1, Fig.5 EFS holdout sans subsampling)
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

from config_v3 import (
    FREQ_DICT,
    CH_NAMES,
    N_EEG,
    SUBJECT_LIST_ORDERED,
    SUBJECT_LABELS,
    CLASSIFICATION_GROUPS,
)
from utils import load_atomic
from classify import StratifiedLeave2GroupsOut  # identique a la classe d'Arthur

BANDS = list(FREQ_DICT)                 # ['delta','theta','alpha','sigma','beta']
EEG_CH = CH_NAMES[:N_EEG]               # 19 electrodes EEG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--state",     type=str, default="S2")
    p.add_argument("--n-jobs",    type=int, default=1)
    p.add_argument("--only-elec", type=str, default="",
                   help="Ne traiter qu'une electrode (ex 'Fz') pour validation rapide.")
    p.add_argument("--max-features", type=int, default=5)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def load_electrode_data(save_path, state):
    """Retourne per_sub : liste (un array (n_epochs, 5_bandes, 19_elec) par sujet) + labels.

    On charge les 5 bandes, chacune (n_epochs, 19) par sujet, et on empile sur un axe
    'bande' -> (n_epochs, 5, 19). Le slice [:, :, e] donnera les 5 features de
    l'electrode e. On garde TOUTES les epochs (pas de subsampling).
    """
    stages = CLASSIFICATION_GROUPS[state]
    per_sub, labels = [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        band_arrays = []
        ok = True
        for b in BANDS:
            parts = [a for s in stages
                     if (a := load_atomic(save_path, f"psd_{b}", sub_id, s)) is not None]
            if not parts:
                ok = False
                break
            band_arrays.append(np.concatenate(parts, axis=0))   # (n_epochs, 19)
        if not ok:
            continue
        # verif meme n_epochs entre bandes (doit etre le cas : meme extraction)
        n0 = band_arrays[0].shape[0]
        if any(a.shape[0] != n0 for a in band_arrays):
            raise RuntimeError(f"sub-{sub_id}: n_epochs incoherent entre bandes.")
        stacked = np.stack(band_arrays, axis=1)                 # (n_epochs, 5, 19)
        per_sub.append(stacked)
        labels.append(label)
    return per_sub, np.array(labels)


def _process_split(subs, labels, train_subs, test_subs, max_features):
    """Traite UN split externe : EFS niche (mlxtend n_jobs=1) + holdout.

    Fonction top-level (picklable) pour Parallel(prefer='processes'). mlxtend est
    force en n_jobs=1 : on parallelise NOUS-MEMES au niveau des splits externes avec
    des PROCESSUS (prefer='processes'), pas des threads. Sur AMD EPYC, threads + GIL +
    contention BLAS ecroulent les perfs ; processus + BLAS mono-thread par process =
    environ 5-6x plus rapide (cf optimisation classify_matrix). D'ou aussi OMP/BLAS=1
    en env.
    Retourne (best_freqs_split, train_score, test_score).
    """
    x_train = np.concatenate([subs[i] for i in train_subs], axis=0)   # (n_ep, 5)
    y_train = np.concatenate([[labels[i]] * len(subs[i]) for i in train_subs])
    g_train = np.concatenate([[i] * len(subs[i]) for i in train_subs])

    clf = LDA()
    f_select = EFS(
        estimator=clf,
        min_features=1,
        max_features=max_features,
        cv=StratifiedLeave2GroupsOut(),
        n_jobs=1,                      # CRITIQUE : pas de nested parallelism
        print_progress=False,
    )
    f_select = f_select.fit(x_train, y_train, groups=g_train)
    best_idx = list(f_select.best_idx_)
    best_freqs_split = [BANDS[j] for j in best_idx]
    train_score = float(f_select.best_score_)

    x_test = np.concatenate([subs[i] for i in test_subs], axis=0)
    y_test = np.concatenate([[labels[i]] * len(subs[i]) for i in test_subs])
    test_clf = LDA()
    test_clf.fit(x_train[:, best_idx], y_train)
    test_score = float(test_clf.score(x_test[:, best_idx], y_test))

    return best_freqs_split, train_score, test_score


def efs_holdout_one_electrode(per_sub, labels, elec_idx, max_features, n_jobs):
    """EFS niche + holdout pour UNE electrode. Replique EFS_fixed_elec.main().

    Les 324 splits externes sont distribues en PROCESSUS (prefer='processes').
    Retourne dict : test_scores, train_scores, best_freqs, score.
    """
    subs = [s[:, :, elec_idx] for s in per_sub]                 # liste (n_epochs, 5)
    groups_all = np.arange(len(subs))
    splits = list(StratifiedLeave2GroupsOut().split(subs, labels, groups_all))

    out = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_split)(subs, labels, tr, te, max_features)
        for tr, te in splits
    )
    best_freqs = [o[0] for o in out]
    train_scores = np.array([o[1] for o in out])
    test_scores = np.array([o[2] for o in out])

    return {
        "test_scores": test_scores,
        "train_scores": train_scores,
        "best_freqs": best_freqs,
        "score": float(np.mean(test_scores)),
    }


def main():
    args = parse_args()
    t0 = time()

    # Nom de sortie : par electrode en mode --only-elec (job array, evite collision),
    # sinon fichier global toutes electrodes.
    if args.only_elec:
        out = args.out_dir / f"efs_holdout_{args.state}_{args.only_elec}.npz"
    else:
        out = args.out_dir / f"efs_holdout_{args.state}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    per_sub, labels = load_electrode_data(args.save_path, args.state)
    n_hr = int((labels == 1).sum())
    n_lr = int((labels == 0).sum())
    print(f"[{args.state}] sujets : {len(labels)} (HR={n_hr}, LR={n_lr}), "
          f"sans subsampling.")

    if args.only_elec:
        if args.only_elec not in EEG_CH:
            raise ValueError(f"{args.only_elec} pas dans {EEG_CH}")
        elec_list = [args.only_elec]
    else:
        elec_list = EEG_CH

    # Parallelisation AU NIVEAU DES SPLITS avec prefer='processes' (dans
    # efs_holdout_one_electrode). mlxtend force en n_jobs=1 pour eviter le nested
    # parallelism. Les electrodes sont traitees en serie (une electrode = 1 tache
    # SLURM dans le job array ; --only-elec pilote quelle electrode).
    results = {}
    for elec in elec_list:
        te = time()
        e_idx = EEG_CH.index(elec)
        res = efs_holdout_one_electrode(per_sub, labels, e_idx,
                                        args.max_features, args.n_jobs)
        results[elec] = res
        dt = time() - te
        print(f"  {elec:4s} : holdout acc = {res['score']:.4f}  "
              f"(train {res['train_scores'].mean():.4f})  [{dt:.0f}s]")

    # empile pour sauvegarde
    elecs = list(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        electrodes=np.array(elecs),
        bands=np.array(BANDS),
        scores=np.array([results[e]["score"] for e in elecs]),
        test_scores=np.array([results[e]["test_scores"] for e in elecs]),
        train_scores=np.array([results[e]["train_scores"] for e in elecs]),
        # best_freqs : objet (liste de listes variable) -> dtype object
        best_freqs=np.array([results[e]["best_freqs"] for e in elecs], dtype=object),
        state=args.state,
        n_hr=n_hr, n_lr=n_lr,
        subsampling=False,
    )
    print(f"\nSauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()