"""Recompute Fig. 3 (thèse Arthur, chap. 1) : PSD / T-values corrigées / LDA en S2.

Réplique la statistique de groupe d'Arthur (ttest_perm_indep.py + maxstat_pval.py du
repo arthurdehgan/sleep) pour produire les données des trois panneaux de la Fig. 3 :

Ce script produit UNIQUEMENT le panneau T-values (colonne du milieu de la Fig. 3) :
pseudo-t two-sided unpaired, permutation AU NIVEAU SUJET, correction maximum
statistics, par bande de fréquence sur 19 électrodes.
-> réplique ttest_perm_unpaired(..., two_tailed=True, correction="maxstat") d'Arthur.

Les deux autres panneaux sont produits ailleurs :
  - PSD moyen HR vs LR (gauche)   : recompute_psd_spectrum_fig3.py
  - LDA accuracy par électrode (droite) : classif single-feature existants +
    plot_topomap_psd_arthur.py

Fidélité au code d'Arthur (repo public, ttest_perm_indep.py)
------------------------------------------------------------
- t-statistique : scipy.stats.ttest_ind(cond1, cond2, equal_var=False)  (Welch)
- permutation   : on concatène cond1+cond2 (n_sujets_HR + n_sujets_LR lignes), on
  re-split selon un index de permutation (échange de labels AU NIVEAU SUJET). C'est
  déjà le schéma RFX correct (Combrisson & Jerbi 2015) : Arthur permute les sujets
  ici, PAS les epochs (contrairement à son classif LDA). Cf perm_test() d'Arthur.
- maxstat       : pour two-sided, on prend |t|, puis max sur l'axe des comparaisons
  de la distribution de permutation ; la p de chaque comparaison est
  sum(|t_obs| <= max_perm)/n_perm. Cf compute_pvalues(correction="maxstat").
- p-value       : (convention d'Arthur, sans +1 au numérateur ; on garde à
  l'identique pour réplique exacte, le +1 est disponible via --add-one).

Périmètre du maxstat (p<0.001)
------------------------------
--maxstat-scope electrodes : max sur les 19 électrodes, séparément par bande. C'est
                             le schéma LITTERAL du code d'Arthur (ttest_perm_unpaired
                             appelé une fois par bande). DÉFAUT.
--maxstat-scope both       : pool électrodes × bandes (19×5=95), correction unique.
                             Correspond au TEXTE de la thèse p52 ("corrected across
                             electrodes AND frequency bands"), pas à son code publié.

Entrées  : {save_path}/psd_{band}/psd_{band}_s{XX}_S2.npz  (clé "data", shape
           (n_epochs, 19)), features Hann déjà extraites (feat_extract_umap_fooof_v4).
Sorties  : {out_dir}/fig3_ttest_S2.npz  (t-values (5,19), p corrigées (5,19), meta).
           Le panneau PSD (courbe continue) est produit à part par
           recompute_psd_spectrum_fig3.py.

Ce script NE fait AUCUN plot (séparation calcul/visu, convention du repo). Le plot
consommera le .npz.

Usage
-----
    python recompute_ttest_fig3.py \\
        --save-path /scratch/alouis/dream_features_noica_1000hz \\
        --out-dir   /scratch/alouis/dream_features_noica_1000hz_corrected \\
        --state     S2 \\
        --n-perm    10000 \\
        --maxstat-scope electrodes \\
        --n-jobs    $SLURM_CPUS_PER_TASK

Author: recompute pour Alex (réplique Arthur chap.1)
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--state",     type=str, default="S2")
    p.add_argument("--n-perm",    type=int, default=10000)
    p.add_argument("--n-jobs",    type=int, default=1)
    # DEFAUT = 'electrodes' : c'est ce que fait le CODE d'Arthur (ttest_perm_indep.py
    # appelle ttest_perm_unpaired une fois PAR BANDE -> maxstat sur les 19 elec seules).
    # 'both' (pool elec x bandes = 95) correspond au TEXTE de la these p52 ("corrected
    # across electrodes AND frequency bands"), qui n'est PAS ce que fait son code publie.
    # On suit le code par defaut ; 'both' dispo si on veut coller au texte.
    p.add_argument("--maxstat-scope", choices=["electrodes", "both"],
                   default="electrodes",
                   help="'electrodes' (DEFAUT) = max sur 19 elec par bande, LITTERAL "
                        "code Arthur. 'both' = pool elec x bandes (95), texte these p52 "
                        "(pas dans son code).")
    p.add_argument("--add-one", action="store_true", default=False,
                   help="p = (sum+1)/(n_perm+1) au lieu de sum/n_perm (convention +1).")
    p.add_argument("--seed", type=int, default=0,
                   help="Graine du generateur de permutations (reproductibilite).")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


# ─── chargement : une valeur PSD par (sujet, bande, electrode) au niveau SUJET ──

def load_subject_band_means(save_path: Path, state: str):
    """Retourne X[band] de shape (n_sujets, 19) : PSD moyenne sur epochs par sujet.

    Le t-test de groupe d'Arthur est AU NIVEAU SUJET : chaque sujet est UNE
    observation, valant la moyenne de sa PSD (bande, electrode) sur ses epochs du
    stade. On aligne sur SUBJECT_LIST_ORDERED / SUBJECT_LABELS (0=LR, 1=HR).
    """
    stages = CLASSIFICATION_GROUPS[state]
    per_band = {b: [] for b in BANDS}
    labels = []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        ok = True
        means = {}
        for b in BANDS:
            parts = [a for s in stages
                     if (a := load_atomic(save_path, f"psd_{b}", sub_id, s)) is not None]
            if not parts:
                ok = False
                break
            arr = np.concatenate(parts, axis=0)          # (n_epochs, 19)
            means[b] = arr.mean(axis=0)                   # (19,) moyenne sur epochs
        if ok:
            for b in BANDS:
                per_band[b].append(means[b])
            labels.append(label)
    labels = np.array(labels)
    X = {b: np.asarray(per_band[b]) for b in BANDS}       # (n_sujets, 19)
    return X, labels


# ─── t-stat + permutations (réplique ttest_perm_indep.py d'Arthur) ────────────

def _ttest_perm(full_mat, index, equal_var=False):
    """t-stats pour un split de permutation (echange de labels niveau sujet).

    Reproduit _ttest_perm + _generate_conds d'Arthur : on prend les lignes 'index'
    comme cond1, le complement comme cond2, puis ttest_ind Welch, on garde t.
    """
    n = len(full_mat)
    index = list(index)
    comp = list(set(range(n)) - set(index))
    perm = np.vstack((full_mat[index], full_mat[comp]))
    c1, c2 = perm[: len(index)], perm[len(index):]
    return ttest_ind(c1, c2, equal_var=equal_var)[0]      # (n_features,)


def _random_perm_indices(n_samples, n_cond1, n_perm, seed):
    """n_perm tirages aleatoires de n_cond1 indices parmi n_samples (sans le split
    identite). Echange de labels au niveau sujet. Arthur enumere des combinaisons ;
    ici tirage aleatoire (n_perm=10000 >> exact impossible avec 36 sujets)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_perm):
        out.append(rng.choice(n_samples, size=n_cond1, replace=False))
    return out


def ttest_maxstat(cond1, cond2, n_perm, two_tailed, scope_pool, seed, add_one, n_jobs):
    """pseudo-t two-sided + maxstat.

    cond1, cond2 : (n_sub_c1, n_feat), (n_sub_c2, n_feat).
    scope_pool   : si True, le max de permutation est pris sur TOUTES les features
                   passees (pool electrodes x bandes) ; sinon max intra-appel (19 elec).
    Retourne (tval, pval_corr, perm_max_distribution).
    """
    tval = ttest_ind(cond1, cond2, equal_var=False)[0]    # (n_feat,)
    full = np.vstack((cond1, cond2))
    n = len(full)
    idxs = _random_perm_indices(n, len(cond1), n_perm, seed)

    perm_t = Parallel(n_jobs=n_jobs)(
        delayed(_ttest_perm)(full, ix, False) for ix in idxs
    )
    perm_t = np.asarray(perm_t)                            # (n_perm, n_feat)
    stat = np.abs(tval) if two_tailed else tval
    perm_stat = np.abs(perm_t) if two_tailed else perm_t

    perm_max = perm_stat.max(axis=1)                       # (n_perm,) max sur features
    scaling = n_perm
    num = (perm_max[:, None] >= stat[None, :]).sum(axis=0).astype(float)
    if add_one:
        pval = (num + 1.0) / (scaling + 1.0)
    else:
        pval = num / scaling
    return tval, pval, perm_max


# NOTE PANNEAU PSD (colonne gauche de la Fig. 3)
# ------------------------------------------------
# La courbe PSD continue (spectre Welch complet 1-45Hz, moyenne sur electrodes et
# sujets, HR vs LR) N'EST PAS produite ici : les .npz de features ne stockent que la
# moyenne par bande, pas le spectre continu. Elle est produite par le script dedie
# recompute_psd_spectrum_fig3.py, qui reutilise compute_psd_spectrum() de
# feat_extract_umap_fooof_v4.py (meme Welch/Hann/WINDOW). Ce script-ci ne fait QUE la
# statistique (panneau T-values, colonne du milieu).


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time()

    out = args.out_dir / f"fig3_ttest_{args.state}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    X, labels = load_subject_band_means(args.save_path, args.state)
    n_hr = int((labels == 1).sum())
    n_lr = int((labels == 0).sum())
    print(f"[{args.state}] sujets charges : {len(labels)}  (HR={n_hr}, LR={n_lr})")
    if n_hr < 2 or n_lr < 2:
        raise RuntimeError("Pas assez de sujets par groupe.")

    # cond1 = HR (label 1), cond2 = LR (label 0) -> t>0 signifie HR>LR
    hr_idx = np.where(labels == 1)[0]
    lr_idx = np.where(labels == 0)[0]

    tvals, pvals = {}, {}
    if args.maxstat_scope == "electrodes":
        # littéral Arthur : 1 correction maxstat par bande, sur 19 électrodes
        for b in BANDS:
            c1, c2 = X[b][hr_idx], X[b][lr_idx]
            tv, pv, _ = ttest_maxstat(
                c1, c2, args.n_perm, two_tailed=True, scope_pool=False,
                seed=args.seed, add_one=args.add_one, n_jobs=args.n_jobs,
            )
            tvals[b], pvals[b] = tv, pv
    else:
        # scope "both" : pool electrodes x bandes -> une seule distribution maxstat
        # sur les 19*5=95 features. On empile les bandes en colonnes.
        c1 = np.concatenate([X[b][hr_idx] for b in BANDS], axis=1)   # (n_hr, 95)
        c2 = np.concatenate([X[b][lr_idx] for b in BANDS], axis=1)   # (n_lr, 95)
        tv, pv, perm_max = ttest_maxstat(
            c1, c2, args.n_perm, two_tailed=True, scope_pool=True,
            seed=args.seed, add_one=args.add_one, n_jobs=args.n_jobs,
        )
        # re-decoupe en bandes de 19
        for i, b in enumerate(BANDS):
            tvals[b] = tv[i * N_EEG:(i + 1) * N_EEG]
            pvals[b] = pv[i * N_EEG:(i + 1) * N_EEG]

    # nombre d'électrodes significatives par bande (p<0.001)
    print("\n=== T-values corrigees (maxstat, p<0.001) ===")
    for b in BANDS:
        nsig = int((pvals[b] < 0.001).sum())
        print(f"  {b:6s} : {nsig:2d}/19 electrodes sig  "
              f"(t range [{tvals[b].min():+.2f}, {tvals[b].max():+.2f}])")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        bands=np.array(BANDS),
        tvals=np.array([tvals[b] for b in BANDS]),        # (5, 19)
        pvals=np.array([pvals[b] for b in BANDS]),        # (5, 19)
        labels=labels,
        n_hr=n_hr, n_lr=n_lr,
        state=args.state,
        n_perm=args.n_perm,
        maxstat_scope=args.maxstat_scope,
        two_tailed=True,
        add_one=args.add_one,
    )
    print(f"\nSauvegarde : {out}")
    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")


if __name__ == "__main__":
    main()