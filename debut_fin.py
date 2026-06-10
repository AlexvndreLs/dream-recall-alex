import numpy as np
from pathlib import Path


class HypnoDiff:
    def __init__(self, path_a, path_b):
        self.a = self._load(path_a)
        self.b = self._load(path_b)

    @staticmethod
    def _load(filepath):
        with open(filepath) as f:
            return np.array([int(line.strip()) for line in f if line.strip()])

    def compare_shift(self, shift):
        a, b = self.a, self.b
        if shift > 0:
            a, b = a[shift:], b[: len(a[shift:])]
        elif shift < 0:
            b, a = b[-shift:], a[: len(b[-shift:])]
        else:
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
        mismatches = np.where(a != b)[0]
        return {
            "shift": shift,
            "n_compared": len(a),
            "n_mismatches": len(mismatches),
            "first_mismatches": mismatches[:5].tolist(),
            "last_mismatches": mismatches[-5:].tolist(),
        }

    def diagnose(self):
        diff = len(self.a) - len(self.b)
        return {
            "len_a": len(self.a),
            "len_b": len(self.b),
            "len_diff": diff,
            "shift_0": self.compare_shift(0),
            "shift_+1": self.compare_shift(1),
            "shift_-1": self.compare_shift(-1),
        }


if __name__ == "__main__":
    base = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/hypnograms")
    diff = HypnoDiff(base / "hyp_per_s4.txt", base / "hyp_jbe_s4.txt")
    for k, v in diff.diagnose().items():
        print(k, v)
