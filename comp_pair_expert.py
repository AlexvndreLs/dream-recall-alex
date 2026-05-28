#!/usr/bin/env python3
"""
Vérifie l'accord inter-annotateur (per vs jbe) pour tous les sujets.
Usage: python check_interrater_all.py
"""

import numpy as np
import os
import glob


class HypnogramComparator:

    def __init__(self, hypno_path):
        self.hypno_path = hypno_path
        self.artefact_codes = [-1, -2]

    def load_and_epoch(self, filepath, epoch_length=30):
        hyp = np.loadtxt(filepath)
        m, n = 1, 1
        hyp_norm = np.array([])
        while m < len(hyp - epoch_length):
            hyp_norm = np.hstack((hyp_norm, hyp[m])) if hyp_norm.size else hyp[m]
            n += 1
            m += 30
        return hyp_norm

    def compute_agreement(self, a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        valid = ~np.isin(a, self.artefact_codes) & ~np.isin(b, self.artefact_codes)
        av, bv = a[valid], b[valid]
        if len(av) < 50:
            return np.nan, np.nan, 0
        acc = np.sum(av == bv) / len(av)
        kappa = self._cohen_kappa(av.astype(int), bv.astype(int))
        return acc, kappa, len(av)

    def _cohen_kappa(self, a, b):
        stages = sorted(set(a) | set(b))
        conf = np.zeros((len(stages), len(stages)))
        idx = {s: i for i, s in enumerate(stages)}
        for p, j in zip(a, b):
            conf[idx[p]][idx[j]] += 1
        po = np.trace(conf) / conf.sum()
        pe = np.sum(conf.sum(axis=0) * conf.sum(axis=1)) / conf.sum() ** 2
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan

    def get_subjects(self):
        per_files = glob.glob(os.path.join(self.hypno_path, 'hyp_per_s*.txt'))
        return sorted([
            os.path.basename(f).replace('hyp_per_', '').replace('.txt', '')
            for f in per_files
        ])

    def run(self):
        subjects = self.get_subjects()
        results = []

        header = f"{'Sujet':<8} | {'per (ep)':>8} | {'jbe (ep)':>8} | {'Diff':>6} | {'Accord':>8} | {'Kappa':>7} | {'N valid':>8}"
        print(header)
        print("-" * len(header))

        for s in subjects:
            per_file = os.path.join(self.hypno_path, f'hyp_per_{s}.txt')
            jbe_file = os.path.join(self.hypno_path, f'hyp_jbe_{s}.txt')

            if not os.path.exists(jbe_file):
                print(f"{s:<8} | pas de fichier jbe")
                continue

            per = self.load_and_epoch(per_file)
            jbe = self.load_and_epoch(jbe_file)
            diff = len(per) - len(jbe)
            acc, kappa, nv = self.compute_agreement(per, jbe)

            flag = ''
            if abs(diff) > 5:
                flag = f'  << diff {diff} epoques ({diff * 30}s)'
            elif not np.isnan(acc) and acc < 0.65:
                flag = '  << accord faible'

            print(f"{s:<8} | {len(per):>8} | {len(jbe):>8} | {diff:>6} | {100*acc:>7.1f}% | {kappa:>7.3f} | {nv:>8}{flag}")
            results.append((s, diff, acc, kappa, nv))

        suspects = [r for r in results if abs(r[1]) > 5 or (not np.isnan(r[2]) and r[2] < 0.65)]
        print(f"\nSujets suspects ({len(suspects)}):")
        for s, diff, acc, kappa, nv in suspects:
            print(f"  {s}: diff={diff} epoques, accord={100*acc:.1f}%, kappa={kappa:.3f}")


if __name__ == '__main__':
    HYPNO_PATH = '/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/hypnograms/'
    comparator = HypnogramComparator(HYPNO_PATH)
    comparator.run()