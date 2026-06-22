"""Analyse des seuils de rejet ICA sur l'ensemble des sujets.

Objectif : choisir empiriquement la méthode (z-score vs corrélation absolue)
et le seuil de détection EOG, plutôt qu'a priori. Le batch a montré que
le z-score à 2.5 est erratique (0 à 10 rejets selon le sujet) -> on caractérise
le comportement de chaque méthode en fonction du seuil, sur tous les sujets.

Pour chaque sujet, recharge l'ICA déjà fittée (branche ica, Picard) + le raw,
et recalcule SANS re-fitter :
  - le score de corrélation EOG par composante (measure='correlation')
  - le z-score EOG par composante (measure='zscore')
  - la proba/label ICLabel par composante (depuis l'ICA -iclabel)

Note : find_bads_muscle est exclu car MNE 1.12.1 crash sur np.prod quand l'ICA
est rechargée depuis .fif (shapes inhomogènes entre slope/focus/smoothness).
Le muscle n'est pas nécessaire pour trancher la méthode EOG.

Sorties :
  - thresholds_scores.csv   : 1 ligne par (sujet, composante), tous les scores
  - thresholds_curves.png   : courbes nb rejets moyen + variance inter-sujets
  - thresholds_decision.png : 4 panneaux pour trancher méthode + seuil
  - thresholds_reco.csv     : seuil recommandé par méthode (automatique)

Usage :
    python analyze_thresholds.py \
        --bids-path  /home/alouis/scratch/dream_bids \
        --deriv-root /home/alouis/scratch/dream_bids/derivatives \
        --out-dir    ./threshold_analysis --iclabel
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import mne
import mne_bids
from mne_icalabel import label_components

from config_v3 import SUBJECT_IDS, HP_FREQ_ICA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bids-path', type=Path, required=True)
    p.add_argument('--deriv-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=Path("./threshold_analysis"))
    p.add_argument('--iclabel', action='store_true',
                   help="Inclure aussi les probas ICLabel (plus lent)")
    return p.parse_args()


def scores_for_subject(sub_str, bids_path, deriv_root, with_iclabel):
    """Recharge l'ICA + raw d'un sujet, recalcule les scores par composante."""
    ica_path = deriv_root / "ica" / f"sub-{sub_str}_task-sleep_ica.fif"
    if not ica_path.exists():
        print(f"  sub-{sub_str}: ICA absente, skip")
        return None

    raw = mne_bids.read_raw_bids(mne_bids.BIDSPath(
        subject=sub_str, task='sleep', root=bids_path, datatype='eeg'), verbose=False)
    raw.load_data()

    raw_for_ica = raw.copy().drop_channels(['misc1', 'misc2', 'misc3'])
    raw_for_ica.filter(l_freq=HP_FREQ_ICA, h_freq=None, verbose=False)

    # voie horizontale comme dans run_ica
    raw_eog = mne.set_bipolar_reference(
        raw_for_ica, anode='EOG_L', cathode='EOG_R',
        ch_name='EOG_horiz', drop_refs=False, copy=True, verbose=False)
    raw_eog.set_channel_types({'EOG_horiz': 'eog'}, verbose=False)

    ica = mne.preprocessing.read_ica(ica_path)
    eog_ch = ['EOG_L', 'EOG_R', 'EOG_horiz']

    _, sc_corr = ica.find_bads_eog(raw_eog, ch_name=eog_ch, measure='correlation',
                                   threshold=999, verbose=False)
    _, sc_z = ica.find_bads_eog(raw_eog, ch_name=eog_ch, measure='zscore',
                                threshold=999, verbose=False)
    corr = np.abs(np.atleast_2d(np.array(sc_corr))).max(axis=0)
    zsc = np.abs(np.atleast_2d(np.array(sc_z))).max(axis=0)

    n = ica.n_components_

    for name, arr in (("eog_corr", corr), ("eog_zscore", zsc)):
        assert arr.shape == (n,), (
            f"sub-{sub_str}: score '{name}' de forme {arr.shape}, attendu ({n},).")

    # ICLabel optionnel
    icl_label = [''] * n
    icl_proba = [np.nan] * n
    if with_iclabel:
        icl_path = deriv_root / "ica" / f"sub-{sub_str}_task-sleep-iclabel_ica.fif"
        if icl_path.exists():
            ica_icl = mne.preprocessing.read_ica(icl_path)
            raw_lab = raw_for_ica.copy().pick('eeg')
            raw_lab.filter(l_freq=None, h_freq=100.0, verbose=False)
            raw_lab.set_eeg_reference('average', verbose=False)
            ld = label_components(raw_lab, ica_icl, method='iclabel')
            m = min(n, len(ld['labels']))
            for i in range(m):
                icl_label[i] = ld['labels'][i]
                icl_proba[i] = float(ld['y_pred_proba'][i])

    rows = []
    for i in range(n):
        rows.append(dict(
            subject=sub_str, comp=i,
            eog_corr=float(corr[i]), eog_zscore=float(zsc[i]),
            iclabel_label=icl_label[i], iclabel_proba=icl_proba[i],
        ))
    print(f"  sub-{sub_str}: {n} composantes")
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sub_str in SUBJECT_IDS:
        try:
            rows = scores_for_subject(sub_str, args.bids_path, args.deriv_root, args.iclabel)
        except Exception as e:
            print(f"  sub-{sub_str}: ERREUR {e}")
            rows = None
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("Aucune donnée. Vérifier que les ICA existent.")
        return

    csv_path = args.out_dir / "thresholds_scores.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV écrit : {csv_path}  ({len(all_rows)} composantes)")

    subjects = sorted(set(r['subject'] for r in all_rows))
    rows_by_subject = {s: [r for r in all_rows if r['subject'] == s] for s in subjects}

    corr_thr = np.linspace(0.2, 0.9, 36)
    z_thr = np.linspace(1.0, 4.0, 31)

    def rejects_per_subject(score_key, thresh):
        return np.array([
            sum(1 for r in rows_by_subject[s] if r[score_key] > thresh)
            for s in subjects
        ])

    def rejected_comps(score_key, thresh):
        out = []
        for s in subjects:
            out.extend(r for r in rows_by_subject[s] if r[score_key] > thresh)
        return out

    # ── figure 1 : courbes nb rejets ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    mean_c = [rejects_per_subject('eog_corr', t).mean() for t in corr_thr]
    std_c  = [rejects_per_subject('eog_corr', t).std()  for t in corr_thr]
    zero_c = [(rejects_per_subject('eog_corr', t) == 0).mean() * 100 for t in corr_thr]

    axes[0, 0].plot(corr_thr, mean_c, 'o-', label='rejets moyen/sujet')
    axes[0, 0].fill_between(corr_thr,
                            np.array(mean_c) - np.array(std_c),
                            np.array(mean_c) + np.array(std_c),
                            alpha=0.2, label='+-ecart-type')
    axes[0, 0].set_title("EOG correlation absolue")
    axes[0, 0].set_xlabel("seuil"); axes[0, 0].set_ylabel("nb rejets EOG")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(corr_thr, zero_c, 'o-', color='crimson')
    axes[0, 1].set_title("EOG correlation : % sujets a 0 rejet")
    axes[0, 1].set_xlabel("seuil"); axes[0, 1].set_ylabel("% sujets a 0")
    axes[0, 1].grid(alpha=0.3)

    mean_z = [rejects_per_subject('eog_zscore', t).mean() for t in z_thr]
    std_z  = [rejects_per_subject('eog_zscore', t).std()  for t in z_thr]
    zero_z = [(rejects_per_subject('eog_zscore', t) == 0).mean() * 100 for t in z_thr]

    axes[1, 0].plot(z_thr, mean_z, 'o-', label='rejets moyen/sujet')
    axes[1, 0].fill_between(z_thr,
                            np.array(mean_z) - np.array(std_z),
                            np.array(mean_z) + np.array(std_z),
                            alpha=0.2, label='+-ecart-type')
    axes[1, 0].axvline(2.5, ls='--', c='gray', label='seuil actuel 2.5')
    axes[1, 0].set_title("EOG z-score")
    axes[1, 0].set_xlabel("seuil"); axes[1, 0].set_ylabel("nb rejets EOG")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(z_thr, zero_z, 'o-', color='crimson')
    axes[1, 1].axvline(2.5, ls='--', c='gray')
    axes[1, 1].set_title("EOG z-score : % sujets a 0 rejet")
    axes[1, 1].set_xlabel("seuil"); axes[1, 1].set_ylabel("% sujets a 0")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    png_path = args.out_dir / "thresholds_curves.png"
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    print(f"Courbes ecrites : {png_path}")

    # ── figure 2 : decision ──────────────────────────────────────────────────
    has_icl = any(r['iclabel_label'] for r in all_rows)
    icl_reject = {'eye blink', 'muscle artifact'}

    def icl_agreement(score_key, thresh):
        rej = [r for r in rejected_comps(score_key, thresh) if r['iclabel_label']]
        if not rej:
            return np.nan
        return sum(1 for r in rej if r['iclabel_label'] in icl_reject) / len(rej)

    def cv(score_key, thr_arr):
        out = []
        for t in thr_arr:
            c = rejects_per_subject(score_key, t)
            out.append(c.std() / c.mean() if c.mean() > 0 else np.nan)
        return np.array(out)

    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))

    ax2[0, 0].plot(corr_thr, cv('eog_corr', corr_thr), 'o-', label='correlation')
    ax2[0, 0].plot(z_thr / 4.0, cv('eog_zscore', z_thr), 's-', label='z-score (seuil/4)')
    ax2[0, 0].set_title("Stabilite inter-sujets : CV = sigma/mu\n(plus bas = plus stable)")
    ax2[0, 0].set_xlabel("seuil (corr) / seuil/4 (z-score)")
    ax2[0, 0].set_ylabel("CV des rejets/sujet")
    ax2[0, 0].legend(); ax2[0, 0].grid(alpha=0.3)

    if has_icl:
        ax2[0, 1].plot(corr_thr,
                       [icl_agreement('eog_corr', t) for t in corr_thr],
                       'o-', label='correlation')
        ax2[0, 1].plot(z_thr / 4.0,
                       [icl_agreement('eog_zscore', t) for t in z_thr],
                       's-', label='z-score (seuil/4)')
        ax2[0, 1].set_ylim(0, 1.05)
        ax2[0, 1].set_title("Accord ICLabel des composantes rejetees\n(fraction eye/muscle confirmee)")
        ax2[0, 1].set_xlabel("seuil (corr) / seuil/4 (z-score)")
        ax2[0, 1].set_ylabel("fraction confirmee eye/muscle")
        ax2[0, 1].legend(); ax2[0, 1].grid(alpha=0.3)
    else:
        ax2[0, 1].text(0.5, 0.5, "ICLabel absent\n(relancer avec --iclabel)",
                       ha='center', va='center')
        ax2[0, 1].axis('off')

    mat_c = np.array([
        [(np.array([r['eog_corr'] for r in rows_by_subject[s]]) > t).sum()
         for t in corr_thr]
        for s in subjects
    ])
    im = ax2[1, 0].imshow(mat_c, aspect='auto', cmap='viridis',
                          extent=[corr_thr[0], corr_thr[-1], len(subjects), 0])
    ax2[1, 0].set_title("Rejets EOG correlation : sujet x seuil")
    ax2[1, 0].set_xlabel("seuil correlation")
    ax2[1, 0].set_ylabel("sujet (index)")
    fig2.colorbar(im, ax=ax2[1, 0], label="nb rejets")

    ax2[1, 1].hist([r['eog_corr'] for r in all_rows], bins=40, alpha=0.6,
                   label='eog_corr', color='steelblue')
    ax2b = ax2[1, 1].twiny()
    ax2b.hist([r['eog_zscore'] for r in all_rows], bins=40, alpha=0.4,
              label='eog_zscore', color='darkorange')
    ax2[1, 1].set_title("Distribution des scores EOG (toutes composantes)")
    ax2[1, 1].set_xlabel("score correlation (bleu)")
    ax2b.set_xlabel("score z-score (orange)")
    ax2[1, 1].set_ylabel("nb composantes")

    fig2.tight_layout()
    png2 = args.out_dir / "thresholds_decision.png"
    fig2.savefig(png2, dpi=120, bbox_inches='tight')
    print(f"Figures decision ecrites : {png2}")

    # ── reco automatique ─────────────────────────────────────────────────────
    def recommend(score_key, thr_arr):
        best = None
        for t in thr_arr:
            counts = rejects_per_subject(score_key, t)
            if (counts == 0).any():
                continue
            cvv = counts.std() / counts.mean() if counts.mean() > 0 else np.inf
            agr = icl_agreement(score_key, t)
            agr_key = -1.0 if np.isnan(agr) else agr
            key = (cvv, -agr_key)
            if best is None or key < best[0]:
                best = (key, t, counts.mean(), cvv, agr)
        return best

    reco_path = args.out_dir / "thresholds_reco.csv"
    with open(reco_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["methode", "seuil_reco", "rejets_moyen", "CV_inter_sujets",
                    "accord_iclabel", "remarque"])
        for key, thr_arr in (("eog_corr", corr_thr), ("eog_zscore", z_thr)):
            b = recommend(key, thr_arr)
            if b is None:
                w.writerow([key, "n/a", "", "", "",
                            "aucun seuil sans sujet a 0 rejet sur la plage testee"])
            else:
                _, t, m, cvv, agr = b
                w.writerow([key, round(float(t), 3), round(float(m), 2),
                            round(float(cvv), 3),
                            "" if agr < 0 else round(float(agr), 3),
                            "min CV sous contrainte 0-rejet, departage par accord ICLabel"])
    print(f"Reco ecrite : {reco_path}")


if __name__ == '__main__':
    main()
