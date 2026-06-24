"""Compare les modes apériodiques FOOOF 'fixed' vs 'knee' sur les vrais epochs.

But : trancher empiriquement aperiodic_mode pour feat_extract, AVANT de figer
le choix. Sur une plage large (1-45Hz), un spectre qui présente un coude (knee)
mal modélisé par 'fixed' fait surajuster des pics gaussiens fantômes et dégrade
le R² (cf littérature sommeil 2024-2025). Ce script mesure l'écart entre les
deux modes pour décider en connaissance de cause.

Critère de décision (par stade, agrégé sur l'échantillon) :
  - delta_R2 = R2(knee) - R2(fixed)
      petit (< ~0.01)  -> 'fixed' suffit (spectre ~linéaire en log-log)
      grand (> ~0.02)  -> 'knee' nécessaire (coude réel, 'fixed' sous-ajuste)
  - delta_npeaks = n_pics(fixed) - n_pics(knee)
      positif marqué   -> 'fixed' surajoute des pics fantômes (signe du coude)

Lecture : si knee améliore nettement le R2 ET/OU fixed détecte plus de pics,
le coude est réel -> passer feat_extract en aperiodic_mode='knee'. Sinon,
garder 'fixed' (et le documenter, signal ~linéaire sur 1-45Hz).

API specparam 2.0.0rc7 (vérifiée empiriquement sur le cluster Fir) :
  - fg.get_metrics('gof_rsquared') -> list[float], un R² par spectre  ✓
  - fg.get_model(i)                -> SpectralModel individuel         ✓
  - fm.results.params.periodic._fit.shape[0] -> n_peaks               ✓
  Note : itération directe sur fg et get_params('metrics', ...) sont
  cassés dans rc7 — ne pas utiliser.

Sorties :
  <out-dir>/fooof_mode_results.csv       — une ligne par sujet × stade × mode
  <out-dir>/fooof_mode_summary.csv       — moyennes par stade (tableau final)
  <out-dir>/fooof_mode_delta_R2.png      — delta_R2 par stade (barplot + strip)
  <out-dir>/fooof_mode_delta_npeaks.png  — delta_npeaks par stade
  <out-dir>/fooof_mode_r2_dist.png       — distributions R² fixed vs knee
  <out-dir>/fooof_mode_example_fits.png  — exemples de fits superposés (S2, S3)

Usage :
    python compare_fooof_mode.py \\
        --deriv-path /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica \\
        --out-dir    ./fooof_mode_analysis \\
        --max-epochs 40 \\
        --n-jobs     4

    # Sous-ensemble pour test rapide :
    python compare_fooof_mode.py \\
        --deriv-path /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica \\
        --out-dir    ./fooof_mode_analysis \\
        --subjects 01 05 10 19 30 \\
        --max-epochs 40 --n-jobs 4
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, SpectralModel

from config_v3 import ATOMIC_STAGES, FOOOF_FREQ_RANGE, SUBJECT_IDS
from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage,
    compute_psd_spectrum,
    _vhdr,
)

STAGE_ORDER = ["S1", "S2", "S3", "S4", "REM"]
STAGE_COLORS = {"S1": "#7fc97f", "S2": "#beaed4", "S3": "#fdc086",
                "S4": "#f0027f", "REM": "#386cb0"}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True)
    p.add_argument("--out-dir",    type=Path, default=Path("./fooof_mode_analysis"))
    p.add_argument("--subjects",   nargs="+", default=None,
                   help="IDs BIDS (ex: 01 05). Défaut: tous les sujets de config_v3.")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--max-peaks",  type=int, default=8)
    p.add_argument("--n-jobs",     type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────

def fit_both_modes(
    flat_psds: np.ndarray,
    freqs: np.ndarray,
    max_peaks: int,
    n_jobs: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fitte fixed et knee sur (n_spectra, n_freqs).

    Returns dict[mode] -> (r2, n_peaks, ap_exp) par spectre.
    ap_exp = exposant aperiodic (indice 1 en mode fixed, indice 2 en mode knee).
    """
    n = flat_psds.shape[0]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # log10(knee<0) pendant optim
        for mode in ("fixed", "knee"):
            fg = SpectralGroupModel(
                aperiodic_mode=mode, max_n_peaks=max_peaks, verbose=False
            )
            fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=n_jobs)
            r2   = np.array(fg.get_metrics("gof_rsquared"), dtype=float)
            npk  = np.array(
                [fg.get_model(i).results.params.periodic._fit.shape[0] for i in range(n)],
                dtype=int,
            )
            # exposant : col 1 (fixed) ou col 2 (knee, col 0 = offset, col 1 = knee param)
            exp_col = 1 if mode == "fixed" else 2
            ap_exp = np.array(
                [fg.get_model(i).results.params.aperiodic._fit[exp_col] for i in range(n)],
                dtype=float,
            )
            out[mode] = (r2, npk, ap_exp)
    return out


def sample_epochs(
    data: np.ndarray, max_epochs: int, rng: np.random.RandomState
) -> np.ndarray:
    n = data.shape[0]
    if n <= max_epochs:
        return data
    return data[rng.choice(n, size=max_epochs, replace=False)]


# ─── figures ──────────────────────────────────────────────────────────────────

def plot_delta_R2(df_sub: pd.DataFrame, out_dir: Path) -> None:
    """Barplot delta_R2 par stade avec points individuels par sujet."""
    stages = [s for s in STAGE_ORDER if s in df_sub["stage"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, stage in enumerate(stages):
        vals = df_sub[df_sub["stage"] == stage]["delta_R2"].values
        ax.bar(i, vals.mean(), color=STAGE_COLORS[stage], alpha=0.7,
               label=stage, width=0.6)
        ax.scatter(
            np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.18, 0.18, len(vals)),
            vals, color="k", s=12, alpha=0.5, zorder=3,
        )
    ax.axhline(0.02,  color="red",    lw=1.2, ls="--", label="seuil 0.02 (knee nécessaire)")
    ax.axhline(0.01,  color="orange", lw=1.0, ls=":",  label="seuil 0.01")
    ax.axhline(0,     color="gray",   lw=0.8)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("delta R²  (knee − fixed)")
    ax.set_title("delta R² par stade\n(>0.02 → knee nécessaire)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fooof_mode_delta_R2.png", dpi=150)
    plt.close(fig)


def plot_delta_npeaks(df_sub: pd.DataFrame, out_dir: Path) -> None:
    """Barplot delta_npeaks par stade."""
    stages = [s for s in STAGE_ORDER if s in df_sub["stage"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, stage in enumerate(stages):
        vals = df_sub[df_sub["stage"] == stage]["delta_npeaks"].values
        ax.bar(i, vals.mean(), color=STAGE_COLORS[stage], alpha=0.7,
               label=stage, width=0.6)
        ax.scatter(
            np.full(len(vals), i) + np.random.default_rng(1).uniform(-0.18, 0.18, len(vals)),
            vals, color="k", s=12, alpha=0.5, zorder=3,
        )
    ax.axhline(0.8,  color="red",  lw=1.2, ls="--", label="seuil 0.8 (fixed surajoute)")
    ax.axhline(0,    color="gray", lw=0.8)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("delta npeaks  (fixed − knee)")
    ax.set_title("delta npeaks par stade\n(>0.8 → fixed surajoute des pics fantômes)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fooof_mode_delta_npeaks.png", dpi=150)
    plt.close(fig)


def plot_r2_distributions(df_raw: pd.DataFrame, out_dir: Path) -> None:
    """Distributions R² fixed vs knee par stade (violin)."""
    stages = [s for s in STAGE_ORDER if s in df_raw["stage"].unique()]
    fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 4), sharey=True)
    for ax, stage in zip(axes, stages):
        sub = df_raw[df_raw["stage"] == stage]
        data_fixed = sub["r2_fixed"].values
        data_knee  = sub["r2_knee"].values
        vp = ax.violinplot([data_fixed, data_knee], positions=[0, 1],
                           showmedians=True, widths=0.6)
        vp["bodies"][0].set_facecolor(STAGE_COLORS[stage])
        vp["bodies"][1].set_facecolor("lightblue")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fixed", "knee"], fontsize=8)
        ax.set_title(stage)
    axes[0].set_ylabel("R²")
    fig.suptitle("Distribution R² par stade — fixed vs knee", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fooof_mode_r2_dist.png", dpi=150)
    plt.close(fig)


def plot_example_fits(
    flat_psds: np.ndarray,
    freqs: np.ndarray,
    max_peaks: int,
    n_jobs: int,
    stage: str,
    sub_id: str,
    out_dir: Path,
    n_examples: int = 6,
) -> None:
    """Superpose fixed et knee sur quelques spectres individuels.

    API specparam 2.0.0rc7 :
      - fm.results.model.modeled_spectrum -> array du fit complet
      - R² extrait via fg.get_metrics("gof_rsquared")[j] (seule API stable)
    """
    n = min(n_examples, flat_psds.shape[0])
    idx = np.linspace(0, flat_psds.shape[0] - 1, n, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for mode, color, ls in [("fixed", "tab:blue", "-"), ("knee", "tab:orange", "--")]:
            fg = SpectralGroupModel(
                aperiodic_mode=mode, max_n_peaks=max_peaks, verbose=False
            )
            fg.fit(freqs, flat_psds[idx], freq_range=FOOOF_FREQ_RANGE, n_jobs=n_jobs)
            r2_all = fg.get_metrics("gof_rsquared")   # list[float], un par spectre
            for j in range(n):
                fm  = fg.get_model(j)
                ax  = axes[j]
                if mode == "fixed":   # spectre réel une seule fois
                    ax.plot(freqs, flat_psds[idx[j]], "k", lw=1.2, alpha=0.7,
                            label="spectre")
                ax.plot(freqs, fm.results.model.modeled_spectrum,   # ← API rc7
                        color=color, lw=1.5, ls=ls,
                        label=f"{mode} R²={r2_all[j]:.3f}")
                ax.set_title(f"spectre {idx[j]}", fontsize=8)
                ax.legend(fontsize=6)

    fig.suptitle(f"Exemples fits fixed vs knee — {stage} sub-{sub_id}", fontsize=10)
    fig.tight_layout()
    fname = out_dir / f"fooof_mode_example_fits_{stage}_sub{sub_id}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"    → figure exemples : {fname.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects if args.subjects is not None else SUBJECT_IDS
    rng      = np.random.RandomState(args.seed)

    # rows_raw : une ligne par sujet × stade (moyennes sur les spectres du sujet)
    rows_raw: list[dict] = []
    # exemple_fits : on en fait un seul (premier sujet, S2 et S3)
    example_done = {"S2": False, "S3": False}

    for sub_id in subjects:
        if not _vhdr(args.deriv_path, sub_id).exists():
            print(f"sub-{sub_id}: absent, skip")
            continue
        print(f"sub-{sub_id}: chargement...")
        try:
            epochs_dict = load_epochs_by_atomic_stage(args.deriv_path, sub_id)
        except Exception as e:
            print(f"  sub-{sub_id}: erreur ({e}), skip")
            continue

        for stage, data in epochs_dict.items():
            sample      = sample_epochs(data, args.max_epochs, rng)
            psds, freqs = compute_psd_spectrum(sample)
            flat        = psds.reshape(-1, psds.shape[-1])

            res = fit_both_modes(flat, freqs, args.max_peaks, args.n_jobs)
            r2f, npf, apf = res["fixed"]
            r2k, npk, apk = res["knee"]

            rows_raw.append(dict(
                subject      = sub_id,
                stage        = stage,
                n_spectra    = len(r2f),
                r2_fixed     = float(r2f.mean()),
                r2_knee      = float(r2k.mean()),
                delta_R2     = float(r2k.mean() - r2f.mean()),
                npeaks_fixed = float(npf.mean()),
                npeaks_knee  = float(npk.mean()),
                delta_npeaks = float(npf.mean() - npk.mean()),
                ap_exp_fixed = float(apf.mean()),
                ap_exp_knee  = float(apk.mean()),
            ))
            print(f"  {stage}: {sample.shape[0]} epochs × 19 = {len(r2f)} spectres | "
                  f"dR²={r2k.mean()-r2f.mean():+.4f}  dnpk={npf.mean()-npk.mean():+.2f}")

            # figure exemples sur le premier sujet disponible, S2 et S3
            if stage in example_done and not example_done[stage]:
                plot_example_fits(
                    flat, freqs, args.max_peaks, args.n_jobs,
                    stage, sub_id, args.out_dir,
                )
                example_done[stage] = True

    if not rows_raw:
        print("Aucun spectre traité.")
        raise SystemExit(1)

    # ─── CSV détaillé ─────────────────────────────────────────────────────────
    df_raw = pd.DataFrame(rows_raw)
    csv_raw = args.out_dir / "fooof_mode_results.csv"
    df_raw.to_csv(csv_raw, index=False, float_format="%.6f")
    print(f"\nCSV détaillé : {csv_raw}")

    # ─── tableau de synthèse par stade ────────────────────────────────────────
    rows_sum = []
    for stage in STAGE_ORDER:
        sub = df_raw[df_raw["stage"] == stage]
        if sub.empty:
            continue
        rows_sum.append(dict(
            stage        = stage,
            n_subjects   = len(sub),
            n_spectra    = int(sub["n_spectra"].sum()),
            R2_fixed     = round(sub["r2_fixed"].mean(),     4),
            R2_knee      = round(sub["r2_knee"].mean(),      4),
            delta_R2     = round(sub["delta_R2"].mean(),     4),
            delta_R2_std = round(sub["delta_R2"].std(),      4),
            npeaks_fixed = round(sub["npeaks_fixed"].mean(), 2),
            npeaks_knee  = round(sub["npeaks_knee"].mean(),  2),
            delta_npeaks = round(sub["delta_npeaks"].mean(), 2),
            ap_exp_fixed = round(sub["ap_exp_fixed"].mean(), 3),
            ap_exp_knee  = round(sub["ap_exp_knee"].mean(),  3),
        ))

    df_sum = pd.DataFrame(rows_sum)
    csv_sum = args.out_dir / "fooof_mode_summary.csv"
    df_sum.to_csv(csv_sum, index=False)

    print("\n" + "=" * 78)
    print("COMPARAISON FIXED vs KNEE — synthèse par stade")
    print("=" * 78)
    print(df_sum.to_string(index=False))

    mean_dR2 = df_sum["delta_R2"].mean()
    mean_dNP = df_sum["delta_npeaks"].mean()
    print("\n" + "-" * 78)
    print(f"delta_R2 moyen     = {mean_dR2:+.4f}  (knee − fixed ; >0 = knee meilleur)")
    print(f"delta_npeaks moyen = {mean_dNP:+.2f}   (fixed − knee ; >0 = fixed surajoute)")
    print("-" * 78)
    if mean_dR2 > 0.02 or mean_dNP > 0.8:
        verdict = "VERDICT : coude réel détecté -> passer aperiodic_mode='knee' dans config_v3.py."
    elif mean_dR2 < 0.01 and abs(mean_dNP) < 0.5:
        verdict = "VERDICT : spectre ~linéaire sur 1-45Hz -> garder 'fixed' (à documenter)."
    else:
        verdict = "VERDICT : zone grise -> inspecter fooof_mode_example_fits_*.png avant de trancher."
    print(verdict)

    # écrire le verdict dans le CSV summary
    df_sum.attrs["verdict"] = verdict
    with open(args.out_dir / "fooof_mode_verdict.txt", "w") as f:
        f.write(f"delta_R2 moyen     = {mean_dR2:+.4f}\n")
        f.write(f"delta_npeaks moyen = {mean_dNP:+.2f}\n")
        f.write(f"{verdict}\n")

    # ─── figures ──────────────────────────────────────────────────────────────
    print("\nGénération des figures...")
    plot_delta_R2(df_raw, args.out_dir)
    plot_delta_npeaks(df_raw, args.out_dir)
    plot_r2_distributions(df_raw, args.out_dir)
    print("Figures sauvées dans", args.out_dir)
    print("Terminé.")
