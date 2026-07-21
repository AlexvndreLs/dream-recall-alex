"""Recompute Fig. 5 (these Arthur chap.1) - AGREGATION ROI (post-traitement).

Fusionne les resultats par electrode (scripts 1 et 2) en donnees par ROI, pretes pour
le plot de la Fig. 5 (pie charts gauche + barplot droit). Replique la logique
d'agregation de visu_piecharts_fselect.py d'Arthur.

Regle d'agregation (verifiee dans visu_piecharts_fselect.py + super_count d'Arthur)
----------------------------------------------------------------------------------
PIE CHARTS (gauche) : selection rate des bandes.
  - Par electrode : compter, sur les 324 splits, combien de fois chaque bande est
    selectionnee par l'EFS (super_count sur best_freqs). Une bande selectionnee dans
    300/324 splits -> count=300.
  - Par ROI : SOMMER les counts des electrodes du ROI (Arthur ligne 38). Le pie montre
    la repartition des 5 bandes. Ce n'est PAS base sur p-values ni accuracies, juste
    la frequence de selection.
  - Filtre SR 25% (legende Fig.5) : seules les COMBINAISONS de bandes depassant 25% de
    taux de selection sont "colorees". On calcule aussi le selection rate des
    combinaisons (sous-ensembles complets), pas seulement des bandes isolees, pour
    permettre au plot d'appliquer ce seuil.

BARPLOT (droite) : accuracy holdout par ROI.
  - Moyenne des test_scores (accuracy holdout non biaisee) des electrodes du ROI,
    ponderee par le nombre de splits (identique : 324/electrode -> moyenne simple).

SEUIL p<0.001 (ligne pointillee du barplot) :
  - Les p-values viennent du script 2 (permutations par split, par electrode). On
    agrege par ROI en prenant la fraction de splits significatifs, et on reporte aussi
    la p-value mediane par ROI. Le seuil p<0.001 est trace par le plot. NB : Arthur
    n'explicite pas une unique p-value par ROI ; on fournit les deux vues (fraction
    sig + mediane) pour que le plot/la note tranchent.

Entrees : {in_dir}/efs_holdout_{state}_{elec}.npz  (19 fichiers, script 1)
          {in_dir}/efs_perms_{state}_{elec}.npz     (19 fichiers, script 2, optionnel)
Sorties : {in_dir}/fig5_roi_{state}.npz
          Par ROI : band_counts (5,), band_rates (5,), combo_rates (dict),
          holdout_acc, frac_sig, p_median. + meta.

Usage
-----
    python aggregate_roi_fig5.py \
        --in-dir /scratch/alouis/dream_features_noica_1000hz_corrected/fig5_recompute \
        --state  S2 --sr-threshold 0.25

Author: recompute pour Alex (replique Arthur chap.1, Fig.5 agregation ROI)
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from config_v3 import FREQ_DICT, CH_NAMES, N_EEG

BANDS = list(FREQ_DICT)
EEG_CH = CH_NAMES[:N_EEG]

# Mapping electrode -> ROI (10-20, identique a classify_multifeature.py).
ROIS = {
    "prefrontal":      ["Fp1", "Fp2"],
    "fronto-central":  ["Fz", "F3", "F4", "FC1", "FC2"],
    "temporal":        ["T3", "T4"],
    "centro-parietal": ["Cz", "C3", "C4", "CP1", "CP2", "Pz", "P3", "P4"],
    "occipital":       ["O1", "O2"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Dossier des efs_holdout_/efs_perms_ par electrode.")
    p.add_argument("--state",  type=str, default="S2")
    p.add_argument("--sr-threshold", type=float, default=0.25,
                   help="Seuil de selection rate pour colorer une combinaison (Fig.5).")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def load_electrode_results(in_dir, state, elec):
    """Charge (best_freqs, test_scores) du script 1 pour une electrode. None si absent."""
    f = in_dir / f"efs_holdout_{state}_{elec}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {
        "best_freqs": list(d["best_freqs"][0]),   # 324 sous-ensembles
        "test_scores": np.asarray(d["test_scores"][0]),
    }


def load_electrode_pvalues(in_dir, state, elec):
    """Charge pvalues du script 2 pour une electrode. None si absent (perms pas encore faites)."""
    f = in_dir / f"efs_perms_{state}_{elec}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return np.asarray(d["pvalues"])


def main():
    args = parse_args()

    out = args.in_dir / f"fig5_roi_{args.state}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} existe deja (--overwrite pour recalculer).")
        return

    # charge tout par electrode
    per_elec = {}
    missing = []
    for elec in EEG_CH:
        res = load_electrode_results(args.in_dir, args.state, elec)
        if res is None:
            missing.append(elec)
            continue
        res["pvalues"] = load_electrode_pvalues(args.in_dir, args.state, elec)
        per_elec[elec] = res
    if missing:
        print(f"ATTENTION electrodes manquantes (script 1 non lance ?) : {missing}")
    if not per_elec:
        raise RuntimeError("Aucun resultat electrode trouve.")

    has_perms = all(per_elec[e]["pvalues"] is not None for e in per_elec)
    if not has_perms:
        print("NB : p-values absentes pour certaines electrodes -> frac_sig/p_median "
              "partiels (script 2 pas encore complet).")

    # comptes par electrode : bandes isolees + combinaisons completes
    band_count_elec = {}     # elec -> Counter(bande -> n splits ou selectionnee)
    combo_count_elec = {}    # elec -> Counter(tuple(bandes triees) -> n splits)
    n_splits_elec = {}
    for elec, res in per_elec.items():
        bc = Counter()
        cc = Counter()
        for subset in res["best_freqs"]:
            for b in subset:
                bc[b] += 1
            cc[tuple(sorted(subset))] += 1
        band_count_elec[elec] = bc
        combo_count_elec[elec] = cc
        n_splits_elec[elec] = len(res["best_freqs"])

    # agregation par ROI
    roi_out = {}
    print(f"\n=== Fig.5 agregation ROI [{args.state}] ===")
    for roi, elecs in ROIS.items():
        elecs_present = [e for e in elecs if e in per_elec]
        if not elecs_present:
            print(f"  {roi:16s} : aucune electrode dispo, ignore.")
            continue

        # PIE : somme des counts de bandes sur les electrodes du ROI (Arthur ligne 38)
        band_counts = np.array([
            sum(band_count_elec[e].get(b, 0) for e in elecs_present) for b in BANDS
        ], dtype=float)
        total_splits = sum(n_splits_elec[e] for e in elecs_present)
        band_rates = band_counts / total_splits    # taux de selection par bande

        # selection rate des COMBINAISONS (pour le filtre SR 25%)
        combo_counts = Counter()
        for e in elecs_present:
            combo_counts.update(combo_count_elec[e])
        combo_rates = {combo: cnt / total_splits for combo, cnt in combo_counts.items()}
        combo_above = {c: r for c, r in combo_rates.items() if r >= args.sr_threshold}

        # BARPLOT : accuracy holdout moyenne (tous splits, toutes elec du ROI)
        all_scores = np.concatenate([per_elec[e]["test_scores"] for e in elecs_present])
        holdout_acc = float(all_scores.mean())

        # p-values agregees (si dispo)
        if all(per_elec[e]["pvalues"] is not None for e in elecs_present):
            all_p = np.concatenate([per_elec[e]["pvalues"] for e in elecs_present])
            frac_sig = float((all_p < 0.001).mean())
            p_median = float(np.median(all_p))
        else:
            frac_sig = np.nan
            p_median = np.nan

        roi_out[roi] = {
            "band_counts": band_counts,
            "band_rates": band_rates,
            "combo_rates": combo_rates,
            "combo_above_threshold": combo_above,
            "holdout_acc": holdout_acc,
            "frac_sig": frac_sig,
            "p_median": p_median,
            "n_elec": len(elecs_present),
        }
        top_band = BANDS[int(np.argmax(band_counts))]
        print(f"  {roi:16s} : acc={holdout_acc:.4f}  top_band={top_band:6s}  "
              f"combos>{int(args.sr_threshold*100)}%={len(combo_above)}  "
              f"frac_sig={frac_sig if np.isnan(frac_sig) else round(frac_sig,3)}")

    # sauvegarde (dict imbrique -> dtype object)
    np.savez(
        out,
        rois=np.array(list(roi_out)),
        bands=np.array(BANDS),
        roi_data=np.array([roi_out], dtype=object),
        state=args.state,
        sr_threshold=args.sr_threshold,
        has_perms=has_perms,
    )
    print(f"\nSauvegarde : {out}")


if __name__ == "__main__":
    main()
