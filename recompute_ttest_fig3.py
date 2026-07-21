"""Recompute Fig. 3 (these Arthur, chap. 1) : panneau T-values corrigees, en S2.

REPLIQUE EXACTE d'Arthur (ttest.py + ttest_perm_indep.py du repo arthurdehgan/sleep).

Ce script produit UNIQUEMENT le panneau T-values (colonne du milieu de la Fig. 3).
Les deux autres panneaux sont produits ailleurs :
  - PSD moyen HR vs LR (gauche)         : recompute_psd_spectrum_fig3.py
  - LDA accuracy par electrode (droite) : classif single-feature + plot_topomap_psd_arthur.py

======================= NIVEAU EPOCH (FFX), PAS NIVEAU SUJET ====================
Le ttest d'Arthur (ttest.py) empile TOUTES les epochs de tous les sujets d'un
groupe en un seul vecteur, puis teste HR-epochs vs LR-epochs. Le bloc qui
moyennerait par sujet est COMMENTE dans son code :
    HR = np.concatenate([psd.flatten() for psd in HR])   # <- toutes epochs
    # for i in range(len(HR)): HR[i] = HR[i].mean()       # <- COMMENTE

Consequence : n = milliers d'epochs (pas 18 sujets), les t explosent, presque
tout devient significatif. C'est le biais FFX (fixed-effects) que le reste du
projet DREAM corrige. La version RFX (1 valeur/sujet, permutation niveau sujet)
donne ~0 electrode sig en S2 : c'est le resultat statistiquement correct, mais ce
script vise la REPLIQUE d'Arthur -> FFX par defaut. Utiliser --level subject pour
la version RFX correcte (comparaison).
================================================================================

Fidelite au code d'Arthur
-------------------------
- z-score : ttest.py d'Arthur charge des fichiers "zscore_psd", MAIS aucun script du
  repo public ne genere ce z-score (compute_psd.py et compute_psd_bins.py sauvent la
  PSD BRUTE, sans z-score ni log ; le generateur zscore_psd n'a jamais ete commite).
  Le z-score est donc irrecuperable. Ce n'est pas bloquant : le t de Welch est
  INVARIANT au rescaling par electrode, donc PSD brute et z-score global par
  electrode donnent des t-values IDENTIQUES. DEFAUT = PSD brute (--zscore none), qui
  repart directement des features extraites sans transformation inventee. L'option
  --zscore global existe pour tracabilite (equivalente). Le z-score PAR SUJET a ete
  teste et ECARTE : il annule les differences de moyenne entre groupes (t=0).
- t-statistique : scipy.stats.ttest_ind(HR_epochs, LR_epochs, equal_var=False), Welch.
- permutation : NIVEAU EPOCH. On concatene toutes les epochs HR + LR, on re-split
  selon des sous-ensembles d'indices d'epochs (perm_test + _combinations d'Arthur).
- maxstat : |t| si two_tailed, puis max sur les 19 electrodes de la distribution de
  permutation. Arthur appelle le ttest une fois par (stade, bande) -> maxstat sur 19
  electrodes seulement (pas sur les bandes). C'est le defaut.
- p-value : sum(|t_obs| <= max_perm)/n_perm, sans +1 (convention d'Arthur).
- exclusions : Arthur exclut le sujet 10 (artefact FC2) et n'a que 17 HR. Reproduit
  via --drop-subjects (defaut : aucun ; passer "10" pour coller a Arthur).

Entrees : {save_path}/psd_{band}/psd_{band}_s{XX}_S2.npz (cle "data", (n_epochs, 19)).
Sorties : {out_dir}/fig3_ttest_{state}.npz (t-values (5,19), p corrigees (5,19), meta).

Ne fait AUCUN plot (separation calcul/visu). Le plot consommera le .npz.

Usage
-----
    python recompute_ttest_fig3.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz \
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_corrected/fig3_recompute \
        --state     S2 --n-perm 9999 --level epoch --zscore none \
        --n-jobs    $SLURM_CPUS_PER_TASK

Author: recompute pour Alex (replique Arthur chap.1, FFX)
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
                   help="9999 = valeur d'Arthur (ttest.py).")
    p.add_argument("--n-jobs",    type=int, default=1)
    p.add_argument("--level", choices=["epoch", "subject"], default="epoch",
                   help="'epoch' (DEFAUT) = FFX, replique Arthur (toutes epochs "
                        "empilees, permutation niveau epoch). 'subject' = RFX correct "
                        "(1 valeur/sujet, permutation niveau sujet).")
    p.add_argument("--zscore", choices=["none", "global"], default="none",
                   help="'none' (DEFAUT) = PSD brute, telle qu'extraite. Aucun z-score "
                        "n'existe dans le code PSD public d'Arthur (le fichier "
                        "'zscore_psd' de ttest.py est genere par un script non commite). "
                        "'global' = z-score par electrode sur le pool complet : "
                        "NUMERIQUEMENT EQUIVALENT a 'none' pour le t (Welch invariant au "
                        "rescaling par colonne), fourni pour tracabilite. Le z-score PAR "
                        "SUJET a ete ecarte : il annule les differences de moyenne entre "
                        "groupes (t=0), verifie empiriquement.")
    p.add_argument("--maxstat-scope", choices=["electrodes", "both"],
                   default="electrodes",
                   help="'electrodes' (DEFAUT) = max sur 19 elec par bande (code "
                        "Arthur). 'both' = pool elec x bandes (texte these p52).")
    p.add_argument("--drop-subjects", type=str, default="",
                   help="IDs sujets a exclure, separes par virgule (ex '10' pour "
                        "coller a Arthur qui retire le sujet 10 / artefact FC2).")
    p.add_argument("--seed", type=int, default=0)
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


def build_conditions(per_band_epochs, labels, level, do_zscore):
    """cond1 (HR), cond2 (LR) par bande.

    level='epoch'   : toutes epochs empilees -> (n_epochs_tot, 19) (FFX).
    level='subject' : moyenne par sujet      -> (n_sujets, 19) (RFX).
    """
    conds = {}
    hr_mask = labels == 1
    lr_mask = labels == 0
    for b in BANDS:
        subs = per_band_epochs[b]
        if do_zscore:
            subs = zscore_global_per_electrode(subs)
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
    """n_perm sous-ensembles aleatoires de n_cond1 indices parmi n_samples.

    Enumeration exhaustive impossible au niveau epoch -> echantillonnage aleatoire,
    comportement d'Arthur des que n_perm < n_comb (cas systematique ici)."""
    rng = np.random.RandomState(seed)
    return [rng.choice(n_samples, size=n_cond1, replace=False) for _ in range(n_perm)]


def ttest_maxstat(cond1, cond2, n_perm, two_tailed, seed, n_jobs):
    tval = ttest_ind(cond1, cond2, equal_var=False)[0]
    full = np.vstack((cond1, cond2))
    idxs = _perm_indices(len(full), len(cond1), n_perm, seed)
    perm_t = Parallel(n_jobs=n_jobs)(delayed(_ttest_perm)(full, ix) for ix in idxs)
    perm_t = np.asarray(perm_t)
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

    drop_ids = {s.strip() for s in args.drop_subjects.split(",") if s.strip()}
    per_band_epochs, labels = load_subject_epochs(args.save_path, args.state, drop_ids)
    n_hr = int((labels == 1).sum())
    n_lr = int((labels == 0).sum())
    do_zscore = args.zscore == "global"
    print(f"[{args.state}] sujets : {len(labels)} (HR={n_hr}, LR={n_lr}) | "
          f"level={args.level} | zscore={args.zscore} | "
          f"drop={sorted(drop_ids) or 'aucun'}")
    if n_hr < 2 or n_lr < 2:
        raise RuntimeError("Pas assez de sujets par groupe.")

    conds = build_conditions(per_band_epochs, labels, args.level, do_zscore)
    if args.level == "epoch":
        n1 = conds[BANDS[0]][0].shape[0]
        n2 = conds[BANDS[0]][1].shape[0]
        print(f"  niveau epoch : {n1} epochs HR vs {n2} epochs LR (n total={n1+n2})")

    tvals, pvals = {}, {}
    if args.maxstat_scope == "electrodes":
        for b in BANDS:
            c1, c2 = conds[b]
            tv, pv = ttest_maxstat(c1, c2, args.n_perm, True, args.seed, args.n_jobs)
            tvals[b], pvals[b] = tv, pv
    else:
        c1 = np.concatenate([conds[b][0] for b in BANDS], axis=1)
        c2 = np.concatenate([conds[b][1] for b in BANDS], axis=1)
        tv, pv = ttest_maxstat(c1, c2, args.n_perm, True, args.seed, args.n_jobs)
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
    )
    print(f"\nSauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()
