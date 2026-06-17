import sys
from pathlib import Path
from collections import Counter
import numpy as np


class HypnogramAligner:
    def __init__(self, filepath, epoch_len=30):
        self.filepath = Path(filepath)
        self.epoch_len = epoch_len
        self.hypno = self._load()

    def _load(self):
        with open(self.filepath) as f:
            return np.array([int(line.strip()) for line in f if line.strip()])

    def transition_indices(self):
        return np.where(np.diff(self.hypno) != 0)[0] + 1

    def offset_distribution(self):
        offsets = self.transition_indices() % self.epoch_len
        return Counter(offsets.tolist())

    def report(self):
        return {
            "file": self.filepath.name,
            "n_samples": len(self.hypno),
            "length_remainder": len(self.hypno) % self.epoch_len,
            "n_transitions": len(self.transition_indices()),
            "offset_distribution": self.offset_distribution(),
        }


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*"
    for f in sorted(directory.glob(pattern)):
        print(HypnogramAligner(f).report())
