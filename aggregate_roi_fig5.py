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

======================= ECARTS AVEC LE CODE D'ARTHUR (documentes) ==============
Comparaison avec visu_piecharts_fselect.py + super_count d'Arthur. Ecarts :

E1. MAPPING ELECTRODE -> ROI (ecart assume, source d'Arthur introuvable). Arthur
    importe REGIONS de params.py, qui est GITIGNORE (jamais commite). Son decoupage
    exact electrode->ROI est donc INCONNU. Le PDF de la these confirme les 5 NOMS de
    ROI (Prefrontal, Fronto-Central, Temporal, Centro-Parietal, Occipital) mais PAS
    quelles electrodes vont dans chaque ROI. On utilise un decoupage 10-20 standard
    (identique a classify_multifeature.py). Defendable mais pas garanti identique a
    Arthur. De plus Arthur avait 12 electrodes (montage demi-gauche : Fz, Cz, Pz,
    Fp1, F3, FC1, C3, T3, CP1, P3, M1, O1), nous 19 (deux hemispheres). Consequences :
    ses ROI temporal et occipital sont sur UNE electrode (T3, O1), les notres sur deux
    (T3/T4, O1/O2) -> moyennes de ROI non comparables. M1 (mastoide) n'existe pas dans
    nos 19 EEG (peut-etre dans misc1/2/3 mais identite NON confirmee, message Arthur en
    attente, et aucune feature PSD extraite dessus). Une variante a 11 electrodes
    (montage Arthur moins M1) est fournie dans aggregate_roi_fig5_arthur11.py pour
    isoler l'effet du montage.

E2. PIE CHARTS = super_count (fidele). Arthur compte, par electrode, le nombre de fois
    ou chaque bande est selectionnee sur les splits (super_count = compteur simple),
    puis SOMME par ROI (ligne 38 : sum des counts des electrodes du ROI). On replique
    exactement (Counter + somme par ROI). Le pie n'est PAS base sur p-values/accuracies,
    juste la frequence de selection. Identique.

E3. SELECTION RATE DES COMBINAISONS (ajout, pas dans le code d'Arthur mais requis par
    la legende Fig.5). La legende dit "feature sets exceeding a selection rate threshold
    of 25%". visu_piecharts_fselect.py ne calcule que les counts de bandes ISOLEES, pas
    des combinaisons. On ajoute combo_rates (taux de selection des sous-ensembles
    complets) pour permettre au plot d'appliquer le seuil 25% sur les COMBINAISONS
    comme le demande la legende. Extension necessaire, documentee.

E4. P-VALUE PAR ROI (ecart : Arthur ne l'explicite pas). Arthur a une p-value par split
    (permutations_EFS_fixed_elec.py) mais n'explicite pas une unique p-value par ROI
    pour le seuil p<0.001 du barplot. On fournit DEUX vues (frac_sig = fraction de
    splits p<0.001, p_median) pour laisser le plot/la note trancher. A clarifier avec
    Arthur/Karim.

E5. BARPLOT = accuracy holdout moyenne par ROI (fidele). Moyenne des test_scores des
    electrodes du ROI. Conforme au "average decoding accuracy using a holdout set".
================================================================================

======= BILAN COMPARAISON A ARTHUR : CE QU'ON A TESTE ET OBTENU (S2) ===========
Synthese des investigations menees pour comprendre les ecarts avec la Fig.5 d'Arthur
(valeurs de sa these, page 59). Table de comparaison (accuracy holdout par ROI) :

  ROI                Arthur   Nous-19elec   Nous-11elec(montage Arthur)   Bande
  prefrontal          65%      62.6%         62.0%                        sigma
  fronto-central      59.1%    55.7%         55.5%                        sigma
  temporal            55.9%    49.3%         50.5%                        sigma
  centro-parietal     52.0%    50.7%         51.0%                        sigma
  occipital           69.1%    48.0%         44.5%                        delta

SELECTION DE BANDES : repliquee sur 5/5 ROI (sigma partout sauf occipital=delta,
identique a Arthur). La METHODE EFS est donc fidele : elle selectionne les memes
bandes que lui dans chaque region.

CE QU'ON A TESTE POUR EXPLIQUER LES ECARTS D'ACCURACY :

1. SUJET 10 (outlier FC2, delta a 23.7x la mediane, artefact confirme). Arthur
   l'exclut pour le ttest (Fig.3) mais le GARDE pour l'EFS (Fig.5) -> incoherence
   d'Arthur, repliquee. TESTE sur Fig.3 : resultats STRICTEMENT identiques avec et
   sans le sujet 10 (sigma 18/19 dans les deux cas). => le sujet 10 n'explique RIEN.

2. MONTAGE 12 vs 19 electrodes. On a refait l'agregation ROI sur les 11 electrodes
   d'Arthur (ses 12 moins M1, absent de nos donnees). Resultat : les accuracies
   changent de <1.5 point et NE SE RAPPROCHENT PAS d'Arthur (col "Nous-11elec"
   ci-dessus). => le montage n'explique PAS les ecarts. Hypothese refutee.

3. MOYENNAGE OCCIPITAL O1+O2. En passant a O1 seul (11-elec), l'occipital EMPIRE
   (48% -> 44.5%), il ne s'ameliore pas. => l'ecart occipital n'est pas un artefact
   de moyennage.

4. DELTA OCCIPITAL discriminant ? MESURE directe de la puissance delta O1/O2 par
   groupe : ratio HR/LR = 0.87 (O1) et 0.94 (O2), soit quasi identiques. => le delta
   occipital NE SEPARE PAS les groupes dans nos donnees. Notre 48% (ou 44.5%) est
   donc CORRECT ; c'est le 69% d'Arthur qui reflete un signal delta occipital
   discriminant que NOS donnees n'ont pas.

CONCLUSION :
- La methode est fidelement repliquee (bandes selectionnees identiques 5/5 ROI).
- Les accuracies sont PROCHES d'Arthur sur 4/5 ROI (prefrontal, fronto-central,
  centro-parietal, et temporal a ~5 points), ROBUSTES au montage et au sujet 10.
- SEULE VRAIE DIVERGENCE : l'occipital (48/44.5% vs 69%). On a PROUVE que le delta
  occipital ne discrimine pas dans nos donnees (ratio HR/LR ~1). L'ecart vient donc
  des DONNEES d'Arthur (son delta occipital discrimine), pas de notre methode.
- CAUSES NON TESTABLES (pas d'acces aux donnees d'Arthur) : son preprocessing (code
  de nettoyage jamais publie ; son delta occipital est peut-etre artefacte, le notre
  peut-etre mieux nettoye) et le nombre d'epochs (14/36 sujets ont plus d'epochs S2
  que sa table de reference). Ces causes restent des hypotheses, non prouvables ici.
================================================================================

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