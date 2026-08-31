"""P-value deflation factor (subject permutation vs epoch permutation).

Reads pairs of {combo}.npz (Random-effects, subject) and {combo}_epochperm.npz
(Fixed-effects, epoch) and calculates the ratio p_RFX / p_FFX.

Two families of files co-exist:
  - matrix (cosp_*, cov_*) : key 'pval', scalar, one p-value per combo
  - vector (psd_*)         : key 'pvals', vector (19,), one p-value per electrode
                             -> we take the minimum across electrodes

Outputs: median and mean of the ratio, by family and overall.
A histogram figure is produced, coloring the two families separately.

Note on interpretation: a portion of FFX p-values saturate at the floor
1/(n_perm+1). The calculated factor is therefore a LOWER BOUND of the true factor.

Usage:
    python3 plot_deflation_full.py \
        --results /scratch/alouis/dream_features_noica_1000hz/results \
        --out plot_perm_explication/deflation_hist_full.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUFFIX = "_epochperm.npz"


def load_pval(path):
    """Returns (p, family) or (None, None).

    family = 'matrix' if scalar key 'pval', 'vector' if vector key 'pvals'
    (in this case we return the minimum across electrodes).
    """
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[skip] {os.path.basename(path)} : {e}")
        return None, None
    keys = list(d.keys())
    if "pvals" in keys:
        pv = np.asarray(d["pvals"], dtype=float).ravel()
        if pv.size == 0:
            return None, None
        return float(np.min(pv)), "vector"
    if "pval" in keys:
        return float(np.asarray(d["pval"], dtype=float).ravel()[0]), "matrix"
    print(f"[skip] {os.path.basename(path)} : neither 'pval' nor 'pvals'")
    return None, None


def collect(results):
    rows = []
    for ef in sorted(glob.glob(os.path.join(results, "*" + SUFFIX))):
        combo = os.path.basename(ef)[: -len(SUFFIX)]
        rf = os.path.join(results, combo + ".npz")
        if not os.path.exists(rf):
            print(f"[skip] {combo} : no matching subject file")
            continue
        pf, fam_f = load_pval(ef)
        pr, fam_r = load_pval(rf)
        if pf is None or pr is None:
            continue
        if fam_f != fam_r:
            print(f"[skip] {combo} : inconsistent families ({fam_f} vs {fam_r})")
            continue
        rows.append((combo, fam_f, pf, pr))
    return rows


def summarize(name, ratios):
    if len(ratios) == 0:
        print(f"  {name:10s} : no combos")
        return
    r = np.asarray(ratios, dtype=float)
    print(
        f"  {name:10s} : n={len(r):3d} | median={np.median(r):7.1f}x | "
        f"mean={np.mean(r):8.1f}x | min={r.min():6.1f}x | max={r.max():8.1f}x"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/scratch/alouis/dream_features_noica_1000hz/results")
    p.add_argument("--out", default="plot_perm_explication/deflation_hist_full.png")
    p.add_argument("--floor", type=float, default=None,
                   help="Floor for zero p-values (default: smallest non-zero p-value observed).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = collect(args.results)
    if not rows:
        raise SystemExit(f"No usable pairs in {args.results}.")

    combos = [r[0] for r in rows]
    fams = np.array([r[1] for r in rows])
    p_ffx = np.array([r[2] for r in rows], dtype=float)
    p_rfx = np.array([r[3] for r in rows], dtype=float)

    if args.floor is not None:
        floor = args.floor
    else:
        nz = np.concatenate([p_ffx[p_ffx > 0], p_rfx[p_rfx > 0]])
        floor = float(nz.min()) if nz.size else 1e-4
    n_sat = int(np.sum(p_ffx <= floor))
    p_ffx_c = np.clip(p_ffx, floor, 1.0)
    p_rfx_c = np.clip(p_rfx, floor, 1.0)

    ratio = p_rfx_c / p_ffx_c

    m_mask = fams == "matrix"
    v_mask = fams == "vector"

    print(f"\n=== Deflation factor p(subject) / p(epoch) ===")
    print(f"floor applied : {floor:g} | p(epoch) saturated at floor : "
          f"{n_sat}/{len(ratio)}")
    summarize("matrices", ratio[m_mask])
    summarize("vectors", ratio[v_mask])
    summarize("overall", ratio)
    print("\nReminder: saturated p(epoch) values make these factors LOWER BOUNDS.\n")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = max(0.5, ratio.min() * 0.8)
    hi = ratio.max() * 1.3
    bins = np.logspace(np.log10(lo), np.log10(hi), 26)

    ax.hist(
        [ratio[v_mask], ratio[m_mask]],
        bins=bins,
        stacked=True,
        color=["#c0392b", "#2c7fb8"],
        edgecolor="white",
        label=[f"spectral features (n={int(v_mask.sum())})",
               f"matrix features (n={int(m_mask.sum())})"],
    )

    med = float(np.median(ratio))
    mean = float(np.mean(ratio))
    ax.axvline(med, color="black", linestyle="--", linewidth=1.6,
               label=f"median = {med:.0f}x")
    ax.axvline(mean, color="black", linestyle="-.", linewidth=1.4,
               label=f"mean = {mean:.0f}x")
    ax.axvline(1.0, color="#777777", linestyle=":", linewidth=1.2,
               label="no deflation")

    ax.set_xscale("log")
    ax.set_xlabel("deflation factor  p(subject) / p(epoch)", fontsize=11)
    ax.set_ylabel("number of feature x stage combinations", fontsize=11)
    ax.set_title(
        "By how much epoch permutation deflates p-values\n"
        f"n = {len(ratio)} combinations (lower bound: "
        f"{n_sat} p-values saturated at floor)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[OK] figure saved : {args.out}")


if __name__ == "__main__":
    main()