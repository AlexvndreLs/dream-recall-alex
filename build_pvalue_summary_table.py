"""Construit un tableau de synthese : pour chaque feature x etat, 6 colonnes
de p-values = {non corrigee, maxstat arthur, maxstat pooled (nous)}
                x {perm subject, perm epoch}.

Ne modifie/n'ecrase RIEN : lecture seule sur results/*.npz et
results/*_epochperm.npz. Ecrit un seul CSV de synthese.

Limite documentee (cf compute_maxstat_correction.py) : le mode "arthur"
(max-stat sur electrodes seulement) n'a pas de sens pour les features
MATRICIELLES (cov, cosp_*) -- une seule mesure, pas d'electrodes. Ces
lignes affichent N/A dans les colonnes arthur, jamais un chiffre invente.

Familles pour la correction "pooled" (notre methode, assumption a valider) :
  - matrix      : cov + cosp_delta/theta/alpha/sigma/beta (6 tests/etat)
  - psd_classic : psd_delta/theta/alpha/sigma/beta x 19 electrodes (95 tests/etat)
  - isolee      : toute autre feature vectorielle, seule sur ses 19 electrodes
                  (pooled == arthur dans ce cas, une seule cle)

Si results/{key}_{state}_epochperm.npz n'existe pas encore (jobs epoch en
cours), les colonnes epoch affichent PENDING plutot que planter ou halluciner
un chiffre.

Usage:
    python build_pvalue_summary_table.py --save-path /scratch/alouis/dream_features_noica_1000hz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["S2", "SWS", "NREM", "REM"]

MATRIX_FAMILY = ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma", "cosp_beta"]
PSD_CLASSIC_FAMILY = ["psd_delta", "psd_theta", "psd_alpha", "psd_sigma", "psd_beta"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Defaut: {save-path}/results/pvalue_summary_table.csv")
    return p.parse_args()


def _load(results_dir: Path, key: str, state: str, suffix: str = ""):
    f = results_dir / f"{key}_{state}{suffix}.npz"
    if not f.exists():
        return None
    return np.load(f, allow_pickle=True)


def is_matrix_feature(key: str) -> bool:
    return key == "cov" or key.startswith("cosp_")


def uncorrected_pval(d, is_matrix: bool):
    """p non corrigee + (electrode si vectoriel, sinon None) + accuracy."""
    if d is None:
        return None, None, None
    if is_matrix:
        return float(d["pval"]), None, float(d["acc_mean"])
    pvals = np.array(d["pvals"])
    acc = np.array(d["acc_mean"])
    ch_names = d["ch_names"] if "ch_names" in d else [str(i) for i in range(len(pvals))]
    best = int(np.argmax(acc))  # meilleure electrode = accuracy max (rang preserve par la correction)
    return float(pvals[best]), str(ch_names[best]), float(acc[best])


def arthur_maxstat_pval(d, is_matrix: bool, best_idx: int | None):
    """Mode arthur : max-stat sur electrodes SEULEMENT, une cle a la fois.
    N/A pour les features matricielles (pas d'electrodes, cf limite documentee)."""
    if is_matrix:
        return None  # N/A explicite, jamais invente
    if d is None or "perm_accs" not in d:
        return None
    perm_accs = np.array(d["perm_accs"])
    acc = np.array(d["acc_mean"])
    n_perm = perm_accs.shape[0]
    null_max = perm_accs.max(axis=1)
    pvals_corr = (np.sum(null_max[:, None] >= acc[None, :], axis=0) + 1) / (n_perm + 1)
    return float(pvals_corr[best_idx])


def pooled_family_pvals(results_dir: Path, keys: list, state: str, suffix: str):
    """Mode pooled (nous) : empile perm_accs de `keys`, un seul pool.
    Retourne dict key -> (pval_corrige_meilleure_electrode_ou_matrice, best_idx_ou_None).
    None si un membre de la famille manque (famille incomplete -> pas de pooling partiel)."""
    loaded = {}
    for key in keys:
        d = _load(results_dir, key, state, suffix)
        if d is None or "perm_accs" not in d:
            return None  # famille incomplete : PENDING pour toute la famille (pas de pooling partiel silencieux)
        loaded[key] = d

    n_perms = {k: len(d["perm_accs"]) for k, d in loaded.items()}
    if len(set(n_perms.values())) > 1:
        return None
    n_perm = next(iter(n_perms.values()))

    perm_pool, real_pool = [], []
    index_map = {}
    col = 0
    for key, d in loaded.items():
        perm_accs = np.array(d["perm_accs"])
        acc = np.array(d["acc_mean"])
        if perm_accs.ndim == 1:  # matriciel : 1 seul test
            perm_pool.append(perm_accs.reshape(-1, 1))
            real_pool.append(float(acc))
            index_map[key] = (col, 0)
            col += 1
        else:  # vectoriel : n_elec tests, on ne garde que la meilleure electrode pour la table
            best = int(np.argmax(acc))
            perm_pool.append(perm_accs[:, best:best + 1])
            real_pool.append(float(acc[best]))
            index_map[key] = (col, 0)
            col += 1

    perm_matrix = np.concatenate(perm_pool, axis=1)
    real_values = np.array(real_pool)
    null_max = perm_matrix.max(axis=1)
    pvals_corr = (np.sum(null_max[:, None] >= real_values[None, :], axis=0) + 1) / (n_perm + 1)

    return {key: float(pvals_corr[idx]) for key, (idx, _) in index_map.items()}


def isolated_pooled_pval(d, is_matrix: bool, best_idx: int | None):
    """Feature isolee (pas dans une famille) : pooled == arthur pour le vectoriel
    (une seule cle -> meme calcul), et pour le matriciel pooled seul = pas de
    max-stat possible (1 seul test, rien a comparer) -> None (N/A)."""
    if is_matrix:
        return None
    return arthur_maxstat_pval(d, is_matrix, best_idx)


if __name__ == "__main__":
    args = parse_args()
    results_dir = args.save_path / "results"
    out_csv = args.out_csv or (results_dir / "pvalue_summary_table.csv")

    all_keys = set()
    for f in results_dir.glob("*.npz"):
        name = f.stem
        if name.endswith("_epochperm"):
            continue
        for state in STATES:
            if name.endswith(f"_{state}"):
                all_keys.add(name[: -(len(state) + 1)])
    all_keys = sorted(all_keys)

    rows = []
    for key in all_keys:
        is_mat = is_matrix_feature(key)
        family = (MATRIX_FAMILY if key in MATRIX_FAMILY else
                  PSD_CLASSIC_FAMILY if key in PSD_CLASSIC_FAMILY else None)

        for state in STATES:
            d_subj = _load(results_dir, key, state, "")
            d_epoch = _load(results_dir, key, state, "_epochperm")

            p_raw_subj, elec, acc = uncorrected_pval(d_subj, is_mat)
            p_raw_epoch, _, _ = uncorrected_pval(d_epoch, is_mat)

            best_idx = None
            if not is_mat and d_subj is not None:
                best_idx = int(np.argmax(np.array(d_subj["acc_mean"])))

            p_arthur_subj  = arthur_maxstat_pval(d_subj, is_mat, best_idx)
            p_arthur_epoch = arthur_maxstat_pval(d_epoch, is_mat, best_idx) if d_epoch is not None else "PENDING"

            if family is not None:
                pooled_subj_all = pooled_family_pvals(results_dir, family, state, "")
                pooled_epoch_all = pooled_family_pvals(results_dir, family, state, "_epochperm")
                p_pooled_subj  = pooled_subj_all.get(key) if pooled_subj_all is not None else "PENDING"
                p_pooled_epoch = pooled_epoch_all.get(key) if pooled_epoch_all is not None else "PENDING"
            else:
                p_pooled_subj  = isolated_pooled_pval(d_subj, is_mat, best_idx)
                p_pooled_epoch = isolated_pooled_pval(d_epoch, is_mat, best_idx) if d_epoch is not None else "PENDING"

            if d_subj is None:
                continue  # combo pas encore calculee du tout, on ne fabrique pas de ligne vide

            rows.append(dict(
                feature=key, state=state, is_matrix=is_mat,
                best_electrode=elec, accuracy_pct=round(acc * 100, 2) if acc else None,
                p_non_corrige_subject=p_raw_subj,
                p_non_corrige_epoch=p_raw_epoch if d_epoch is not None else "PENDING",
                p_maxstat_arthur_subject=p_arthur_subj if p_arthur_subj is not None else "N/A (matriciel)",
                p_maxstat_arthur_epoch=p_arthur_epoch if p_arthur_epoch is not None and p_arthur_epoch != "PENDING" else ("N/A (matriciel)" if is_mat and d_epoch is not None else p_arthur_epoch),
                p_maxstat_pooled_subject=p_pooled_subj if p_pooled_subj is not None else "N/A (1 seul test)",
                p_maxstat_pooled_epoch=p_pooled_epoch if p_pooled_epoch is not None else "N/A (1 seul test)",
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Ecrit : {out_csv}  ({len(df)} lignes)")
    print(df.to_string(index=False))
