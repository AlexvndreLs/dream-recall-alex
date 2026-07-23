"""Recompute Fig. 3 (these Arthur, chap. 1) : panneau T-values corrigees, en S2.

REPLIQUE EXACTE d'Arthur (ttest.py + ttest_perm_indep.py du repo arthurdehgan/sleep).

Ce script produit UNIQUEMENT le panneau T-values (colonne du milieu de la Fig. 3).
Les deux autres panneaux sont produits ailleurs :
  - PSD moyen HR vs LR (gauche)         : recompute_psd_spectrum_fig3.py
  - LDA accuracy par electrode (droite) : classif single-feature + plot_topomap_psd_arthur.py

==================== CE QUE FAIT REELLEMENT ARTHUR (TEXTE DE LA THESE) ==========
!!! CORRECTION IMPORTANTE (07/2026) : une version anterieure de cet en-tete affirmait
qu'Arthur faisait un t-test NIVEAU EPOCH (FFX) avec un z-score PAR SUJET. C'ETAIT
FAUX. C'etait une deduction a partir de son CODE PUBLIC (ttest.py), qui est une
version de travail non finale ("no longer updated, there might be bugs" en tete de
ses fichiers). Le TEXTE DE LA THESE (correctedthesis, chap.1, section stats) decrit
la vraie methode, et elle est differente :

  Thèse, methode du t-test (Fig.3) :
    "subject-wise PSD values were averaged across epochs. The t-test was then
     performed [...] To account for multiple comparisons across electrodes,
     maximum statistics and exhaustive permutations (1000 permutations) were
     employed at a significance level of p < 0.001."
    + correction "across electrodes AND frequency bands".

  Donc le t-test d'Arthur est :
    - NIVEAU SUJET (RFX) : on moyenne les epochs par sujet AVANT le test (18 HR
      vs 18 LR), PAS un empilement de milliers d'epochs. -> --level subject.
    - SANS z-score : le chapitre ne decrit AUCUN z-score des PSD sommeil (les
      seules mentions de normalisation du chapitre concernent ImageNet, hors sujet).
      -> --zscore none.
    - pseudo-t permutationnel two-sided unpaired, 1000 perms. -> --n-perm 1000.
    - maxstat sur electrodes ET bandes. -> --maxstat-scope both.

  Resultat annonce PAR ARTHUR lui-meme (these) :
    "the T-test did not reveal any statistically significant differences in
     spectral power between HR and LR (p > 0.05)".
  Autrement dit : AUCUNE etoile dans la colonne t-values, non pas a cause d'un
  z-score qui ecrase l'effet, mais simplement parce qu'au niveau sujet (n=18 vs 18)
  avec correction maxstat, l'effet PSD univarie n'atteint pas la significativite.
  C'est un resultat correct, pas un artefact. Verifie empiriquement : notre repro
  --level subject donne 0/19 partout avec des t modeles (sigma max ~+2.5), i.e. une
  carte structuree SANS etoile, exactement comme sa Fig.3.

  CONSEQUENCE sur l'argumentaire du projet : le contraste FFX vs RFX est valable
  pour la CLASSIFICATION riemannienne (ou Arthur fait bien du FFX/epoch discutable),
  mais PAS pour ce t-test : pour le t-test, Arthur est DEJA en RFX. Ne pas melanger
  les deux en presentation.

COMMANDE POUR REPRODUIRE FIDELEMENT ARTHUR (Fig.3, colonne t-values) :
    --level subject --zscore none --maxstat-scope both --n-perm 1000 --drop-subjects 10
================================================================================

Options z-score du script (tracabilite, PAS ce que fait Arthur)
---------------------------------------------------------------
- --zscore none (DEFAUT) : PSD brute. C'est la methode de la these (aucun z-score).
- --zscore global : z-score par electrode sur le pool. NUMERIQUEMENT EQUIVALENT a
  'none' pour le t de Welch (invariant au rescaling par colonne). Tracabilite.
- --zscore subject : z-score PAR SUJET. Annule les differences de moyenne inter-
  groupes (t=0 exact). Ce mode a ete ajoute quand on croyait (a tort) qu'Arthur
  z-scorait par sujet ; la these ne le confirme PAS. A n'utiliser que pour explorer
  l'effet d'un tel z-score, jamais comme "reproduction d'Arthur".
- t-statistique : scipy.stats.ttest_ind(cond1, cond2, equal_var=False), Welch.
- t-statistique : scipy.stats.ttest_ind(HR_epochs, LR_epochs, equal_var=False), Welch.
- permutation : NIVEAU EPOCH. On concatene toutes les epochs HR + LR, on re-split
  selon des sous-ensembles d'indices d'epochs (perm_test + _combinations d'Arthur).
- maxstat : |t| si two_tailed, puis max sur les 19 electrodes de la distribution de
  permutation. Arthur appelle le ttest une fois par (stade, bande) -> maxstat sur 19
  electrodes seulement (pas sur les bandes). C'est le defaut.
- p-value : sum(|t_obs| <= max_perm)/n_perm, sans +1 (convention d'Arthur).
- exclusions : Arthur exclut le sujet 10 (artefact FC2) et n'a que 17 HR. Reproduit
  via --drop-subjects (defaut : aucun ; passer "10" pour coller a Arthur).

======================= ECARTS AVEC LE CODE D'ARTHUR (documentes) ==============
Comparaison ligne par ligne avec ttest.py + ttest_perm_indep.py d'Arthur. Ecarts :

E1. ECHANTILLONNAGE DES PERMUTATIONS (ecart reel, methodologiquement en notre faveur).
    Arthur : itertools.combinations(range(n), n_cond1) tronque aux n_perm PREMIERES
    combinaisons (_combinations, ligne 303-310). Deterministe mais BIAISE : les
    premieres combinaisons lexicographiques gardent presque toutes les memes
    echantillons ensemble (ex (0,1,2,...) puis (0,1,...,n-1,n+1)), donc ce n'est PAS
    un echantillon uniforme de l'espace des permutations. Au niveau epoch (~10000
    epochs) c'est encore plus biaise.
    Nous : rng.choice(replace=False), tirage ALEATOIRE UNIFORME. Statistiquement
    superieur (echantillonnage non biaise de la distribution nulle), mais differe de
    son code. Impact a verifier empiriquement (sur un vrai effet fort comme sigma/S2,
    les deux convergent car le t observe est loin dans les deux nulls).

E2. perm_t[1:] (ecart mineur). Arthur retire la 1ere permutation (l'identite, qui
    redonne le split original non permute ; ligne 180 : "return perm_t[1:]"). Nous ne
    generons PAS l'identite (tirage aleatoire), donc rien a retirer. Ecart de 1 perm
    sur 9999, impact negligeable sur le p. SOUS-DETAIL : chez Arthur, ce retrait
    affecte aussi le DENOMINATEUR du p-value (scaling = len(perm_t) APRES le [1:], donc
    scaling = n_perm - 1 = 9998). Nous divisons par n_perm exact (9999). Ecart de 1 sur
    9999 au denominateur, negligeable.

E3. EXCLUSION SUJET 10 (ecart reel, controle par --drop-subjects). Arthur exclut le
    sujet 10 pour le ttest UNIQUEMENT (np.delete(X,9) + X[:17] -> 17 HR + 18 LR ;
    ttest.py ligne 31-33). VERIFIE dans nos donnees : sujet 10 = outlier FC2 delta a
    23.7x la mediane (artefact reel confirme). Pour repliquer Arthur : --drop-subjects
    10. NB : Arthur GARDE le sujet 10 pour la Fig.5 (EFS) -> incoherence d'Arthur,
    reproduite (cf recompute_efs_holdout_fig5.py).

Points VERIFIES IDENTIQUES (pas des ecarts) : formule p sum(t_obs<=t_perm)/n_perm
sans +1 ; maxstat = perm_t.max(axis=1) sur 19 elec par bande ; _generate_conds
(vstack index/complement, re-split) ; two_tailed via abs() ; equal_var=False (Welch) ;
niveau epoch (concatenation de toutes les epochs, moyenne par sujet commentee chez lui).

E4. MONTAGE 12 vs 19 ELECTRODES (ecart de donnees). Arthur corrige le maxstat sur SES
    12 electrodes (Fz,Cz,Pz,Fp1,F3,FC1,C3,T3,CP1,P3,M1,O1), nous sur nos 19. Le maxstat
    etant un max sur les electrodes, corriger sur 19 est PLUS severe que sur 12 (plus
    de comparaisons -> seuil plus haut). Donc a effet egal, on aura potentiellement
    MOINS d'electrodes significatives qu'Arthur, uniquement a cause du nombre
    d'electrodes dans la correction. M1 absent de nos donnees (misc non identifies).

E5. BANDES : alpha (8-13) et sigma (11-16) SE CHEVAUCHENT (11-13 Hz communs). C'est le
    decoupage d'Arthur (identique dans le PDF), donc fidele, mais a garder en tete :
    les t-values alpha et sigma ne sont pas independantes sur 11-13 Hz. Non corrige
    (fidelite a Arthur).

E6. BIAIS DE SIGNE DU TWO-TAILED D'ARTHUR (ecart reel, NON reproduit par defaut).
    Dans compute_pvalues (ttest_perm_indep.py), Arthur applique abs() a la
    distribution nulle (perm_t = abs(perm_t)) mais compare la statistique OBSERVEE
    SIGNEE (boucle : if tstat <= t_perm). Comme la nulle est en valeur absolue
    (toujours >= 0), une electrode a effet negatif (t_obs < 0, i.e. HR < LR) verifie
    t_obs <= |t_perm| pour PRESQUE TOUTES les permutations -> p ~ 1. Son "two-tailed"
    ne detecte donc en pratique QUE les effets HR > LR ; tout effet HR < LR est
    invisible. C'est une erreur de signe (le abs() manque sur l'observe), pas une
    convention. NOTRE DEFAUT compare |t_obs| a |t_perm| (two-tailed symetrique,
    correct). Le flag --arthur-pval-bug reproduit son comportement a l'identique,
    UNIQUEMENT pour la section "reproduction" (comparaison figure a figure), jamais
    comme resultat. C'est la cause principale des differences visuelles sur la colonne
    t-values entre sa Fig.3 et la notre (sur ses vraies donnees le z-score par sujet
    met les t a ~0 et masque le bug ; il ne devient visible que sur PSD brute, donc
    dans notre replique).

E6-bis. IMPACT EMPIRIQUE MESURE DU BUG E6 (verifie 07/2026, S2, branche noica
    1000hz, script check_sign_bug_impact.py). Le bug ne peut affecter QUE les
    electrodes a t_obs < 0 ; sur les t positifs les deux methodes sont identiques
    par construction. Resultat : SON IMPACT DEPEND ENTIEREMENT DU NIVEAU DE
    PERMUTATION.

      RFX (--level subject, conforme a la these, 17 HR vs 18 LR, scope both,
      1000 perms) : 38/95 couples (bande, electrode) ont t_obs < 0, mais le t le
      plus negatif est a peine -1.40 (delta, e12). AUCUN effet negatif ne franchit
      p<0.05 ni p<0.001, ni en version correcte ni en version Arthur. Le bug ne
      masque RIEN d'observable. Les deux topomaps sont visuellement identiques.
      -> Sur la Fig.3 telle qu'Arthur la produit, ce bug est SANS CONSEQUENCE.

      FFX (--level epoch, 9999 perms, scope electrodes, mode d'exploration) :
      31/95 couples ont t_obs < 0, avec des magnitudes enormes (jusqu'a -11.2 en
      alpha). Le bug MASQUE alors 12 electrodes a p<0.05 et 10 a p<0.001, toutes
      poussees a p = 1.0000 exactement.

    LECTURE : le bug est une erreur reelle, mais il ne devient destructeur que
    combine a une distribution nulle artificiellement etroite (permutation epoch).
    Au niveau d'inference correct il est inoffensif sur ces donnees. NE PAS
    presenter ce bug comme expliquant les resultats de la Fig.3 d'Arthur : il ne
    les explique pas. Le vrai ecart methodologique porte sur le panneau accuracies
    (FFX chez Arthur, RFX chez nous), pas sur le t-test.

    ATTENTION a ne pas confondre avec l'exclusion du sujet 10 (E3) : celle-ci est
    motivee par un artefact FC2 delta reel (23.7x la mediane), decidee sur la
    qualite des donnees, SANS rapport avec ce bug. Le sujet 10 tirait la carte
    delta vers le negatif (t e12 : -1.54 avec lui, -1.40 sans), ce qui affaiblit
    encore les effets negatifs apres exclusion, mais ne change aucune conclusion.

E6-ter. BUG DE TYPE SUR --drop-subjects (corrige 07/2026). SUBJECT_LIST_ORDERED
    contient des int, alors que drop_ids etait construit comme un set de str :
    `10 in {"10"}` vaut False, donc l'exclusion n'avait JAMAIS lieu, en silence,
    pendant que le log affichait `drop=['10']` comme si elle avait fonctionne.
    Corrige par un cast int() explicite (leve une exception si non castable,
    plutot qu'une exclusion fantome). Symptome de detection : le log annonce 36
    sujets (18 HR / 18 LR) au lieu de 35 (17 HR / 18 LR).
================================================================================

Entrees : {save_path}/psd_{band}/psd_{band}_s{XX}_S2.npz (cle "data", (n_epochs, 19)).
Sorties : {out_dir}/fig3_ttest_{state}.npz (t-values (5,19), p corrigees (5,19), meta).

Ne fait AUCUN plot (separation calcul/visu). Le plot consommera le .npz.

Usage
-----
    # Reproduction FIDELE d'Arthur (Fig.3, d'apres la these) : RFX niveau sujet.
    python recompute_ttest_fig3.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute \
        --state     S2 --n-perm 1000 --level subject --zscore none \
        --maxstat-scope both --drop-subjects 10 \
        --n-jobs    $SLURM_CPUS_PER_TASK

Author: recompute pour Alex (replique Arthur chap.1 ; t-test RFX niveau sujet
        conforme au texte de la these, cf en-tete)
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from config_v3 import (
    FREQ_DICT,
    N_EEG,
    SUBJECT_LIST_ORDERED,
    SUBJECT_LABELS,
    CLASSIFICATION_GROUPS,
)
from utils import load_atomic

BANDS = list(FREQ_DICT)  # ['delta','theta','alpha','sigma','beta']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--state",     type=str, default="S2")
    p.add_argument("--n-perm",    type=int, default=9999,
                   help="DEFAUT 9999. Pour repliquer la these : 1000 (valeur du "
                        "texte pour le t-test).")
    p.add_argument("--n-jobs",    type=int, default=1)
    p.add_argument("--level", choices=["epoch", "subject"], default="epoch",
                   help="ATTENTION : pour REPLIQUER ARTHUR (Fig.3), utiliser "
                        "'subject'. La these decrit un t-test niveau sujet "
                        "('subject-wise PSD averaged across epochs'), PAS niveau "
                        "epoch. 'epoch' (defaut historique) = FFX, n=milliers "
                        "d'epochs, t gonfles : ce N'EST PAS ce que fait Arthur pour "
                        "ce t-test, c'est un mode d'exploration. 'subject' = RFX, "
                        "conforme a la these.")
    p.add_argument("--zscore", choices=["none", "global", "subject"], default="none",
                   help="'none' (DEFAUT, = these : aucun z-score des PSD sommeil). "
                        "'global' = z-score par electrode sur le pool : equivalent "
                        "numerique de 'none' pour le t de Welch, tracabilite. "
                        "'subject' = z-score par sujet : ANNULE l'effet de groupe "
                        "(t=0). Ajoute quand on croyait a tort qu'Arthur z-scorait "
                        "par sujet ; la these ne le confirme PAS. Exploration only.")
    p.add_argument("--maxstat-scope", choices=["electrodes", "both"],
                   default="electrodes",
                   help="Pour repliquer Arthur (Fig.3), utiliser 'both' : la these "
                        "corrige 'across electrodes AND frequency bands'. "
                        "'electrodes' (defaut historique) = max sur 19 elec par bande.")
    p.add_argument("--drop-subjects", type=str, default="",
                   help="IDs sujets a exclure, separes par virgule (ex '10' pour "
                        "coller a Arthur qui retire le sujet 10 / artefact FC2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arthur-pval-bug", action="store_true", default=False,
                   help="Reproduit le biais de signe two-tailed d'Arthur (cf E6) : "
                        "abs() sur la nulle seulement, observe signe. Les effets "
                        "HR<LR deviennent invisibles. UNIQUEMENT pour repliquer sa "
                        "Fig.3 a l'identique, jamais comme resultat.")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def load_subject_epochs(save_path, state, drop_ids):
    """per_band[band] = liste (un array (n_epochs,19) par sujet) + labels.

    On garde TOUTES les epochs (pas de moyenne) : c'est ce qui permet le FFX.
    """
    stages = CLASSIFICATION_GROUPS[state]
    per_band = {b: [] for b in BANDS}
    labels = []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        if sub_id in drop_ids:
            continue
        ok = True
        per_sub = {}
        for b in BANDS:
            parts = [a for s in stages
                     if (a := load_atomic(save_path, f"psd_{b}", sub_id, s)) is not None]
            if not parts:
                ok = False
                break
            per_sub[b] = np.concatenate(parts, axis=0)   # (n_epochs, 19)
        if ok:
            for b in BANDS:
                per_band[b].append(per_sub[b])
            labels.append(label)
    return per_band, np.array(labels)


def zscore_global_per_electrode(epochs_list):
    """z-score PAR ELECTRODE sur le POOL complet (toutes epochs, tous sujets).

    Normalise l'echelle (PSD ~1e-12) sans effacer les differences inter-groupes :
    on calcule mu/sd par colonne electrode sur l'ensemble empile, PUIS on applique
    a chaque sujet. NUMERIQUEMENT EQUIVALENT a pas de z-score pour le t de Welch
    (invariant au rescaling par colonne). A ne PAS confondre avec un z-score par
    sujet, qui lui annule l'effet de groupe (verifie : t=0).
    """
    pool = np.concatenate(epochs_list, axis=0)           # (n_epochs_tot, 19)
    mu = pool.mean(axis=0, keepdims=True)
    sd = pool.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)                       # garde-fou division par 0
    return [(arr - mu) / sd for arr in epochs_list]


def zscore_per_subject(epochs_list):
    """z-score PAR SUJET, par electrode (sur les epochs de chaque sujet).

    REPRODUIT LE zscore_psd D'ARTHUR (d'apres prepare_data, utils.py : zscore par
    sous-matrice sujet). ATTENTION : ce z-score ANNULE les differences de moyenne
    inter-groupes (chaque sujet recentre sur 0) -> le ttest HR vs LR tombe a t~0,
    AUCUNE electrode significative. C'est ce qui explique l'absence d'etoiles chez
    Arthur. A n'utiliser QUE pour reproduire sa Fig.3 non significative.
    """
    out = []
    for arr in epochs_list:
        mu = arr.mean(axis=0, keepdims=True)
        sd = arr.std(axis=0, ddof=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        out.append((arr - mu) / sd)
    return out


def build_conditions(per_band_epochs, labels, level, zscore_mode):
    """cond1 (HR), cond2 (LR) par bande.

    level='epoch'   : toutes epochs empilees -> (n_epochs_tot, 19) (FFX).
    level='subject' : moyenne par sujet      -> (n_sujets, 19) (RFX).
    zscore_mode : 'none' (brut), 'global' (par elec sur pool, equivalent none pour t),
                  'subject' (par sujet, REPRODUIT ARTHUR -> non significatif).
    """
    conds = {}
    hr_mask = labels == 1
    lr_mask = labels == 0
    for b in BANDS:
        subs = per_band_epochs[b]
        if zscore_mode == "global":
            subs = zscore_global_per_electrode(subs)
        elif zscore_mode == "subject":
            subs = zscore_per_subject(subs)
        hr = [subs[i] for i in range(len(subs)) if hr_mask[i]]
        lr = [subs[i] for i in range(len(subs)) if lr_mask[i]]
        if level == "epoch":
            cond1 = np.concatenate(hr, axis=0)
            cond2 = np.concatenate(lr, axis=0)
        else:
            cond1 = np.stack([s.mean(axis=0) for s in hr])
            cond2 = np.stack([s.mean(axis=0) for s in lr])
        conds[b] = (cond1, cond2)
    return conds


def _ttest_perm(full_mat, index):
    n = len(full_mat)
    index = list(index)
    comp = list(set(range(n)) - set(index))
    perm = np.vstack((full_mat[index], full_mat[comp]))
    c1, c2 = perm[: len(index)], perm[len(index):]
    return ttest_ind(c1, c2, equal_var=False)[0]


def _perm_indices(n_samples, n_cond1, n_perm, seed):
    """n_perm sous-ensembles ALEATOIRES de n_cond1 indices parmi n_samples.

    ECART E1 avec Arthur (cf en-tete) : Arthur utilise itertools.combinations tronque
    aux n_perm premieres combinaisons (deterministe mais BIAISE). Nous faisons un
    tirage aleatoire uniforme (rng.choice), statistiquement superieur mais different
    de son code. L'enumeration exhaustive est de toute facon impossible au niveau
    epoch (comb(~10000, ~5000) est astronomique)."""
    rng = np.random.RandomState(seed)
    return [rng.choice(n_samples, size=n_cond1, replace=False) for _ in range(n_perm)]


def ttest_maxstat(cond1, cond2, n_perm, two_tailed, seed, n_jobs,
                  arthur_pval_bug=False):
    """t-test maxstat par permutation, deux modes de p-value two-tailed.

    arthur_pval_bug=False (DEFAUT, correct) : two-tailed symetrique. On compare
        |t_obs| au max sur electrodes de |t_perm|. Un effet fort dans un sens ou
        dans l'autre (HR>LR ou HR<LR) est detecte de facon identique.

    arthur_pval_bug=True (REPLIQUE EXACTE Arthur, cf E6) : reproduit le biais de
        signe de compute_pvalues (ttest_perm_indep.py). Arthur applique abs() a la
        distribution nulle mais PAS a la statistique observee, et compte
        tstat <= t_perm. Consequence : pour une electrode a effet negatif
        (t_obs<0), la condition t_obs <= |t_perm| est vraie pour presque toutes
        les permutations -> p ~ 1, l'effet HR<LR devient invisible. Son two-tailed
        se comporte donc comme un one-tailed HR>LR. A n'utiliser QUE pour
        reproduire sa Fig.3 a l'identique, jamais comme resultat.
    """
    tval = ttest_ind(cond1, cond2, equal_var=False)[0]
    full = np.vstack((cond1, cond2))
    idxs = _perm_indices(len(full), len(cond1), n_perm, seed)
    perm_t = Parallel(n_jobs=n_jobs)(delayed(_ttest_perm)(full, ix) for ix in idxs)
    perm_t = np.asarray(perm_t)

    if arthur_pval_bug and two_tailed:
        # Reproduction fidele du bug : abs() sur la nulle SEULEMENT, observe signe,
        # inegalite tstat <= t_perm (comme la boucle de compute_pvalues d'Arthur).
        perm_max = np.abs(perm_t).max(axis=1)                 # max_elec |t_perm|
        num = (perm_max[:, None] >= tval[None, :]).sum(axis=0).astype(float)
        return tval, num / n_perm

    stat = np.abs(tval) if two_tailed else tval
    perm_stat = np.abs(perm_t) if two_tailed else perm_t
    perm_max = perm_stat.max(axis=1)
    num = (perm_max[:, None] >= stat[None, :]).sum(axis=0).astype(float)
    return tval, num / n_perm


def main():
    args = parse_args()
    t0 = time()

    out = args.out_dir / f"fig3_ttest_{args.state}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    # SUBJECT_LIST_ORDERED contient des int : comparer a des str echoue
    # silencieusement (10 in {"10"} -> False). Cast explicite, erreur si
    # non castable plutot que exclusion fantome.
    drop_ids = {int(s.strip()) for s in args.drop_subjects.split(",") if s.strip()}
    per_band_epochs, labels = load_subject_epochs(args.save_path, args.state, drop_ids)
    n_hr = int((labels == 1).sum())
    n_lr = int((labels == 0).sum())
    print(f"[{args.state}] sujets : {len(labels)} (HR={n_hr}, LR={n_lr}) | "
          f"level={args.level} | zscore={args.zscore} | "
          f"drop={sorted(drop_ids) or 'aucun'}")
    if n_hr < 2 or n_lr < 2:
        raise RuntimeError("Pas assez de sujets par groupe.")

    conds = build_conditions(per_band_epochs, labels, args.level, args.zscore)
    if args.level == "epoch":
        n1 = conds[BANDS[0]][0].shape[0]
        n2 = conds[BANDS[0]][1].shape[0]
        print(f"  niveau epoch : {n1} epochs HR vs {n2} epochs LR (n total={n1+n2})")

    if args.arthur_pval_bug:
        print("  [MODE ARTHUR] biais de signe two-tailed active (E6) : "
              "effets HR<LR invisibles. Replique exacte, PAS un resultat.")

    tvals, pvals = {}, {}
    if args.maxstat_scope == "electrodes":
        for b in BANDS:
            c1, c2 = conds[b]
            tv, pv = ttest_maxstat(c1, c2, args.n_perm, True, args.seed, args.n_jobs,
                                   arthur_pval_bug=args.arthur_pval_bug)
            tvals[b], pvals[b] = tv, pv
    else:
        c1 = np.concatenate([conds[b][0] for b in BANDS], axis=1)
        c2 = np.concatenate([conds[b][1] for b in BANDS], axis=1)
        tv, pv = ttest_maxstat(c1, c2, args.n_perm, True, args.seed, args.n_jobs,
                               arthur_pval_bug=args.arthur_pval_bug)
        for i, b in enumerate(BANDS):
            tvals[b] = tv[i * N_EEG:(i + 1) * N_EEG]
            pvals[b] = pv[i * N_EEG:(i + 1) * N_EEG]

    print("\n=== T-values corrigees (maxstat, p<0.001) ===")
    for b in BANDS:
        nsig = int((pvals[b] < 0.001).sum())
        print(f"  {b:6s} : {nsig:2d}/19 electrodes sig  "
              f"(t range [{tvals[b].min():+.2f}, {tvals[b].max():+.2f}])")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        bands=np.array(BANDS),
        tvals=np.array([tvals[b] for b in BANDS]),
        pvals=np.array([pvals[b] for b in BANDS]),
        labels=labels,
        n_hr=n_hr, n_lr=n_lr,
        state=args.state,
        n_perm=args.n_perm,
        level=args.level,
        zscore=args.zscore,
        maxstat_scope=args.maxstat_scope,
        drop_subjects=sorted(drop_ids),
        two_tailed=True,
        arthur_pval_bug=args.arthur_pval_bug,
    )
    print(f"\nSauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()