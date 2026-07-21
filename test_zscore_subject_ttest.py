"""Test ponctuel : est-ce que le z-score PAR SUJET (comme le zscore_psd d'Arthur,
d'apres prepare_data dans son utils.py) tue la significativite du ttest, expliquant
qu'Arthur n'ait AUCUNE etoile en t-values ?

Compare, en S2, le nombre d'electrodes significatives par bande AVEC vs SANS z-score
par sujet. Si le z-score par sujet fait tomber sigma de 18/19 a ~0/19, on a l'explication.

Usage : python test_zscore_subject_ttest.py --save-path ... --state S2 --n-perm 9999
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind, zscore
from joblib import Parallel, delayed
from config_v3 import FREQ_DICT, SUBJECT_LIST_ORDERED, SUBJECT_LABELS, CLASSIFICATION_GROUPS
from utils import load_atomic

BANDS = list(FREQ_DICT)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--state", default="S2")
    p.add_argument("--n-perm", type=int, default=9999)
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()

def load(save_path, state):
    stages = CLASSIFICATION_GROUPS[state]
    per_band = {b: [] for b in BANDS}; labels = []
    for sid, lab in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        if str(sid) == "10":   # exclu comme Arthur (ttest)
            continue
        ok = True; tmp = {}
        for b in BANDS:
            parts = [a for s in stages if (a := load_atomic(save_path, f"psd_{b}", sid, s)) is not None]
            if not parts: ok = False; break
            tmp[b] = np.concatenate(parts, axis=0)
        if ok:
            for b in BANDS: per_band[b].append(tmp[b])
            labels.append(lab)
    return per_band, np.array(labels)

def ttest_maxstat(c1, c2, n_perm, seed, n_jobs):
    tval = ttest_ind(c1, c2, equal_var=False)[0]
    full = np.vstack((c1, c2)); n = len(full); n1 = len(c1)
    rng = np.random.RandomState(seed)
    idxs = [rng.choice(n, size=n1, replace=False) for _ in range(n_perm)]
    def one(ix):
        comp = list(set(range(n)) - set(ix))
        perm = np.vstack((full[ix], full[comp]))
        return ttest_ind(perm[:n1], perm[n1:], equal_var=False)[0]
    pt = np.abs(np.asarray(Parallel(n_jobs=n_jobs)(delayed(one)(ix) for ix in idxs)))
    pmax = pt.max(axis=1)
    return tval, (pmax[:, None] >= np.abs(tval)[None, :]).sum(axis=0) / n_perm

def run(per_band, labels, do_zscore, n_perm, n_jobs):
    hr_i = np.where(labels == 1)[0]; lr_i = np.where(labels == 0)[0]
    res = {}
    for b in BANDS:
        subs = per_band[b]
        if do_zscore:
            subs = [zscore(a, axis=0) for a in subs]   # z-score PAR SUJET, par electrode
        hr = np.concatenate([subs[i] for i in hr_i], axis=0)
        lr = np.concatenate([subs[i] for i in lr_i], axis=0)
        _, pv = ttest_maxstat(hr, lr, n_perm, 0, n_jobs)
        res[b] = int((pv < 0.001).sum())
    return res

def main():
    a = parse_args()
    pb, lab = load(a.save_path, a.state)
    print(f"[{a.state}] {len(lab)} sujets (sujet 10 exclu)")
    print("\nSANS z-score (notre version):")
    r0 = run(pb, lab, False, a.n_perm, a.n_jobs)
    for b in BANDS: print(f"  {b:6s}: {r0[b]}/19")
    print("\nAVEC z-score PAR SUJET (hypothese Arthur zscore_psd):")
    r1 = run(pb, lab, True, a.n_perm, a.n_jobs)
    for b in BANDS: print(f"  {b:6s}: {r1[b]}/19")
    print("\n=> si la colonne AVEC tombe a ~0 partout, le z-score par sujet explique")
    print("   l'absence d'etoiles chez Arthur.")

if __name__ == "__main__":
    main()