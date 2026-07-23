"""Bootstrap convergence: cumulative mean of accuracies over 1,000 resamples.

DIAGNOSTIC figure, not a result: it verifies that the number of bootstraps is
sufficient to stabilize the estimation; it says nothing about HR vs LR.

To be read with caution: a cumulative mean mechanically converges as 1/sqrt(n),
so a flat curve primarily proves that the arithmetic works. What it usefully
shows is the residual AMPLITUDE of oscillations near the end: if they remain
large compared to the difference between two features being compared, 1,000
bootstraps are not enough to tell them apart.

The grey band represents the cumulative standard error (std/sqrt(n)): it materializes
this expected 1/sqrt(n) behavior. A curve straying out of its own band signals an
issue (non-independent resamples, drift).

Usage:
    python plot_bootstrap_convergence.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir   /scratch/alouis/dream-recall-alex/plot_overlap \
        --features cosp_sigma/S2 cosp_delta/SWS cov/REM psd_sigma/S2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    RESOLUTION,
    band_label,
    is_matrix_key,
    key_color,
    load_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+",
                   default=["cosp_sigma/S2", "cosp_delta/SWS", "cov/REM", "psd_sigma/S2"],
                   help="List of 'feature/state' pairs.")
    return p.parse_args()


def cumulative(scores: np.ndarray):
    """Cumulative mean and cumulative standard error, in %.

    scores: (n_boot,) accuracies, each already averaged across the 324 splits.
    """
    n = np.arange(1, len(scores) + 1)
    cummean = np.cumsum(scores) / n
    # Cumulative std using the König-Huygens formula: avoids looping over n_boot
    cumvar = np.cumsum(scores ** 2) / n - cummean ** 2
    cumse = np.sqrt(np.maximum(cumvar, 0) / n)
    return cummean * 100, cumse * 100


def main() -> None:
    args = parse_args()
    print("=== bootstrap convergence ===")

    pairs = []
    for tok in args.features:
        if "/" not in tok:
            raise SystemExit(f"--features: expected format feature/state, received '{tok}'")
        feat, state = tok.split("/", 1)
        pairs.append((feat.strip(), state.strip()))

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.4),
                             squeeze=False)
    axes = axes[0]

    for ax, (key, state) in zip(axes, pairs):
        d = load_result(args.save_path, key, state)
        if d is None:
            print(f"  missing: {key}_{state}.npz")
            ax.axis("off")
            continue

        scores = np.asarray(d["acc_scores"])
        # For a vector feature, acc_scores is (n_boot, 19): we track the best
        # electrode, consistent with what other figures test.
        ch = None
        if not is_matrix_key(key):
            best = int(np.asarray(d["acc_mean"]).argmax())
            ch = str(d["ch_names"][best]) if "ch_names" in d.files else f"elec {best}"
            scores = scores[:, best]

        cummean, cumse = cumulative(scores)
        n = np.arange(1, len(scores) + 1)

        ax.fill_between(n, cummean - cumse, cummean + cumse,
                        color="0.8", lw=0, label="± cumulative SE")
        ax.plot(n, cummean, color=key_color(key), lw=1.5)
        ax.axhline(cummean[-1], color="k", ls="--", lw=0.8)

        title = f"{band_label(key)}, {state}"
        if ch:
            title += f" ({ch})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cumulative bootstraps")
        ax.set_xscale("log")  # Most of the convergence occurs before n=100
        ax.spines[["top", "right"]].set_visible(False)

        # Residual amplitude: the only truly informative metric here.
        tail = cummean[len(cummean) // 2:]
        ax.text(0.97, 0.06,
                f"final = {cummean[-1]:.2f}%\n"
                f"2nd-half amplitude = {tail.max() - tail.min():.2f} pt",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.8", alpha=0.85))

    axes[0].set_ylabel("Cumulative mean accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Bootstrap convergence (diagnostic)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "bootstrap_convergence.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()