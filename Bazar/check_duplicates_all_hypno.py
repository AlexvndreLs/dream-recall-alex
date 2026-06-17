#!/usr/bin/env python3
"""
Compare tous les fichiers hypnogramme entre eux et detecte les doublons/swaps.
Seuls les fichiers existants sont compares (pas de jbe manquants).
Usage: python check_duplicates.py
"""

import numpy as np
import os
import glob
from itertools import combinations


class HypnogramDuplicateChecker:

    def __init__(self, hypno_path, output_path):
        self.hypno_path = hypno_path
        self.output_path = output_path
        self.artefact_codes = [-1, -2]

    def load_and_epoch(self, filepath, epoch_length=30):
        hyp = np.loadtxt(filepath)
        m = 1
        hyp_norm = []
        while m < len(hyp - epoch_length):
            hyp_norm.append(hyp[m])
            m += 30
        return np.array(hyp_norm)

    def similarity(self, a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        valid = ~np.isin(a, self.artefact_codes) & ~np.isin(b, self.artefact_codes)
        av, bv = a[valid], b[valid]
        if len(av) < 50:
            return np.nan, 0
        return np.sum(av == bv) / len(av), len(av)

    def get_subject(self, name):
        return name.replace('hyp_per_', '').replace('hyp_jbe_', '')

    def load_all(self, log):
        files = sorted(glob.glob(os.path.join(self.hypno_path, 'hyp_*.txt')))
        data = {}
        for f in files:
            name = os.path.basename(f).replace('.txt', '')
            data[name] = self.load_and_epoch(f)
            msg = f"  charge {name}: {len(data[name])} epoques"
            print(msg)
            log.write(msg + '\n')
        return data

    def write_table(self, rows, log):
        col_header = f"{'Fichier A':<22} | {'Fichier B':<22} | {'ep A':>6} | {'ep B':>6} | {'Accord':>8} | {'N valid':>8}"
        separator = "-" * len(col_header)
        for line in [col_header, separator]:
            print(line)
            log.write(line + '\n')
        for name_a, name_b, ep_a, ep_b, acc, nv in rows:
            line = f"{name_a:<22} | {name_b:<22} | {ep_a:>6} | {ep_b:>6} | {100*acc:>7.1f}% | {nv:>8}"
            print(line)
            log.write(line + '\n')

    def run(self):
        with open(self.output_path, 'w') as log:

            header = "Comparaison exhaustive des hypnogrammes\n" + "=" * 80 + "\n"
            print(header)
            log.write(header)

            data = self.load_all(log)
            names = list(data.keys())
            n_pairs = len(names) * (len(names) - 1) // 2

            msg = f"\n{len(names)} fichiers charges, {n_pairs} paires a comparer\n"
            print(msg)
            log.write(msg + '\n')

            same_subject_rows = []
            diff_subject_rows = []

            for name_a, name_b in combinations(names, 2):
                a = data[name_a]
                b = data[name_b]
                acc, nv = self.similarity(a, b)
                if np.isnan(acc):
                    continue

                row = (name_a, name_b, len(a), len(b), acc, nv)
                if self.get_subject(name_a) == self.get_subject(name_b):
                    same_subject_rows.append(row)
                else:
                    diff_subject_rows.append(row)

            # Meme sujet : ordre croissant (accord faible en premier)
            same_subject_rows.sort(key=lambda x: x[4])
            # Sujets differents : ordre decroissant (suspects en premier)
            diff_subject_rows.sort(key=lambda x: -x[4])

            section = "\n--- Paires MEME SUJET (ordre croissant d'accord) ---\n"
            print(section)
            log.write(section + '\n')
            self.write_table(same_subject_rows, log)

            section = "\n--- Paires SUJETS DIFFERENTS (ordre decroissant d'accord) ---\n"
            print(section)
            log.write(section + '\n')
            self.write_table(diff_subject_rows, log)

        print(f"\nResultats sauvegardes dans {self.output_path}")


if __name__ == '__main__':
    HYPNO_PATH = '/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/hypnograms/'
    OUTPUT_PATH = '/home/alouis/dream-recall-alex/hypno_duplicate_check.txt'
    checker = HypnogramDuplicateChecker(HYPNO_PATH, OUTPUT_PATH)
    checker.run()