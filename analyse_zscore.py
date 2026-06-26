"""Analyse z-score du rejet EOG, à partir du CSV de scores déjà calculé.

Important : find_bads_eog renvoie TOUJOURS la corrélation de Pearson comme
score, quel que soit `measure`. La différence z-score vs corrélation est
uniquement la règle de DÉCISION :
  - corrélation : rejette si |corr| > seuil_absolu
  - z-score     : rejette les outliers à `seuil` écarts-types, via un z-scoring
                  itératif PAR SUJET (mne.preprocessing.bads._find_outliers)

Ce script reproduit fidèlement _find_outliers de MNE (max_iter=2, tail=0) sur
les corrélations de chaque sujet, et refait les figures de décision pour la
règle z-score. Aucune relecture des données EEG : tout part du CSV.

Usage (local, instantané) :
    python analyze_zscore.py --scores ./threshold_analysis/thresholds_scores.csv \
                             --out-dir ./threshold_analysis
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import zscore


def find_outliers(X, threshold=3.0, max_iter=2, tail=0):
    """Copie exacte de mne.preprocessing.bads._find_outliers."""
    X = np.asarray(X, dtype=float)
    my_mask = np.zeros(len(X), dtype=bool)
    for _ in range(max_iter):
        Xm = np.ma.masked_array(X, my_mask)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            this_z = np.abs(zscore(Xm))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break
    return np.where(my_mask)[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', type=Path, required=True,
                   help="thresholds_scores.csv produit par analyze_thresholds")
    p.add_argument('--out-dir', type=Path, default=Path("./threshold_analysis"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # corrélations par sujet (eog_corr == eog_zscore dans le CSV : même Pearson)
    corr_by_subject = defaultdict(list)
    icl_by_subject = defaultdict(list)
    for r in csv.DictReader(open(args.scores)):
        corr_by_subject[r['subject']].append(float(r['eog_corr']))
        icl_by_subject[r['subject']].append(r.get('iclabel_label', ''))
    subjects = sorted(corr_by_subject)
    corr_by_subject = {s: np.array(v) for s, v in corr_by_subject.items()}

    # plage de seuils z-score (en écarts-types), comme dans l'analyse corrélation
    z_thr = np.linspace(1.0, 4.0, 31)

    def rejects_z_per_subject(thresh):
        """Nb d'outliers z-score par sujet, à seuil donné (règle MNE)."""
        return np.array([
            len(find_outliers(corr_by_subject[s], threshold=thresh))
            for s in subjects
        ])

    def rejected_comps_z(thresh):
        """(sujet, idx_comp) rejetés au seuil z donné."""
        out = []
        for s in subjects:
            for i in find_outliers(corr_by_subject[s], threshold=thresh):
                out.append((s, int(i)))
        return out

    # ── figure décision z-score ──────────────────────────────────────────────
    icl_reject = {'eye blink', 'muscle artifact'}
    has_icl = any(any(lbl for lbl in v) for v in icl_by_subject.values())

    def icl_agreement_z(thresh):
        rej = rejected_comps_z(thresh)
        rej = [(s, i) for (s, i) in rej if icl_by_subject[s][i]]
        if not rej:
            return np.nan
        ok = sum(1 for (s, i) in rej if icl_by_subject[s][i] in icl_reject)
        return ok / len(rej)

    def cv_z(thr_arr):
        out = []
        for t in thr_arr:
            c = rejects_z_per_subject(t)
            out.append(c.std() / c.mean() if c.mean() > 0 else np.nan)
        return np.array(out)

    mean_z = [rejects_z_per_subject(t).mean() for t in z_thr]
    std_z  = [rejects_z_per_subject(t).std()  for t in z_thr]
    zero_z = [(rejects_z_per_subject(t) == 0).mean() * 100 for t in z_thr]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0, 0].plot(z_thr, mean_z, 'o-', label='rejets moyen/sujet')
    ax[0, 0].fill_between(z_thr,
                          np.array(mean_z) - np.array(std_z),
                          np.array(mean_z) + np.array(std_z),
                          alpha=0.2, label='+-ecart-type')
    ax[0, 0].axvline(2.5, ls='--', c='gray', label='seuil 2.5')
    ax[0, 0].set_title("EOG z-score : rejets par sujet (regle MNE _find_outliers)")
    ax[0, 0].set_xlabel("seuil z (ecarts-types)"); ax[0, 0].set_ylabel("nb rejets")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(z_thr, zero_z, 'o-', color='crimson')
    ax[0, 1].axvline(2.5, ls='--', c='gray')
    ax[0, 1].set_title("EOG z-score : % sujets a 0 rejet")
    ax[0, 1].set_xlabel("seuil z"); ax[0, 1].set_ylabel("% sujets a 0")
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(z_thr, cv_z(z_thr), 'o-', color='teal')
    ax[1, 0].axvline(2.5, ls='--', c='gray')
    ax[1, 0].set_title("EOG z-score : stabilite inter-sujets (CV)\n(plus bas = plus stable)")
    ax[1, 0].set_xlabel("seuil z"); ax[1, 0].set_ylabel("CV des rejets/sujet")
    ax[1, 0].grid(alpha=0.3)

    if has_icl:
        ax[1, 1].plot(z_thr, [icl_agreement_z(t) for t in z_thr], 'o-', color='purple')
        ax[1, 1].axvline(2.5, ls='--', c='gray')
        ax[1, 1].set_ylim(0, 1.05)
        ax[1, 1].set_title("EOG z-score : accord ICLabel (fraction eye/muscle)")
        ax[1, 1].set_xlabel("seuil z"); ax[1, 1].set_ylabel("fraction confirmee")
        ax[1, 1].grid(alpha=0.3)
    else:
        ax[1, 1].text(0.5, 0.5, "ICLabel absent", ha='center', va='center')
        ax[1, 1].axis('off')

    fig.tight_layout()
    png = args.out_dir / "thresholds_zscore_decision.png"
    fig.savefig(png, dpi=120, bbox_inches='tight')
    print(f"Figure z-score ecrite : {png}")

    # reco z-score
    reco = args.out_dir / "thresholds_zscore_reco.csv"
    with open(reco, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["seuil_z", "rejets_moyen", "CV_inter_sujets",
                    "accord_iclabel", "pct_sujets_0"])
        best = None
        for t in z_thr:
            c = rejects_z_per_subject(t)
            if (c == 0).any():
                continue
            cvv = c.std() / c.mean() if c.mean() > 0 else np.inf
            agr = icl_agreement_z(t)
            if best is None or cvv < best[1]:
                best = (t, cvv, c.mean(), agr)
        for t in (2.0, 2.5, 3.0):
            c = rejects_z_per_subject(t)
            cvv = c.std() / c.mean() if c.mean() > 0 else float('nan')
            agr = icl_agreement_z(t)
            w.writerow([t, round(c.mean(), 2), round(cvv, 3),
                        "" if np.isnan(agr) else round(agr, 3),
                        round((c == 0).mean() * 100, 1)])
        if best:
            t, cvv, m, agr = best
            w.writerow([f"opt={round(t,2)}", round(m, 2), round(cvv, 3),
                        "" if (agr is None or np.isnan(agr)) else round(agr, 3),
                        0.0])
    print(f"Reco z-score ecrite : {reco}")


if __name__ == '__main__':
    main()
