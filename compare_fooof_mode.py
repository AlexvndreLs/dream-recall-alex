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

Réutilise load_epochs_by_atomic_stage et compute_psd_spectrum de feat_extract
(pas de duplication de la logique d'épochage / Welch).

API specparam 2.0 :
  - fg.get_params("r_squared") -> (n_spectra,)         [était get_metrics("gof")]
  - fm.n_peaks_                -> int par spectre       [itération sur fg]
  Les deux corrections par rapport à la version initiale du script.

Usage :
    python compare_fooof_mode.py \\
        --deriv-path /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica \\
        --subjects   01 05 10 19 23 \\
        --max-epochs 40 \\
        --max-peaks  8

    # Tous les sujets (plus long, ~30-40 min sur nœud interactif 4 CPUs) :
    python compare_fooof_mode.py \\
        --deriv-path /home/alouis/scratch/dream_bids/derivatives/preprocessed-ica
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from specparam import SpectralGroupModel

from config_v3 import ATOMIC_STAGES, FOOOF_FREQ_RANGE, SUBJECT_IDS
from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage,
    compute_psd_spectrum,
    _vhdr,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True,
                   help="Racine du derivative preprocessed-ica")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="IDs BIDS à échantillonner (ex: 01 05 10). Défaut: tous.")
    p.add_argument("--max-epochs", type=int, default=40,
                   help="Nombre max d'epochs échantillonnés par sujet/stade (défaut: 40)")
    p.add_argument("--max-peaks", type=int, default=8,
                   help="max_n_peaks passé à FOOOF (défaut: 8, comme la littérature sommeil)")
    p.add_argument("--seed", type=int, default=42,
                   help="Graine pour l'échantillonnage des epochs (défaut: 42)")
    return p.parse_args()


# ─── helpers ──────────────────────────────────────────────────────────────────

def count_peaks_per_spectrum(fg: SpectralGroupModel, n_spectra: int) -> np.ndarray:
    """Nombre de pics gaussiens détectés par spectre (specparam 2.0).

    Itère sur le SpectralGroupModel (chaque élément est un SpectralModel
    individuel) et lit fm.n_peaks_. Compatible specparam 2.0.x.
    Ne pas utiliser get_params("peak") + bincount : l'index de spectre
    d'origine n'est pas garanti en col -1 dans toutes les versions 2.0.
    """
    counts = np.zeros(n_spectra, dtype=int)
    for i, fm in enumerate(fg):
        counts[i] = fm.n_peaks_
    return counts


def fit_both_modes(
    flat_psds: np.ndarray, freqs: np.ndarray, max_peaks: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fitte fixed et knee sur les mêmes spectres aplatis (n_spectra, n_freqs).

    Correction specparam 2.0 : get_params("r_squared") remplace get_metrics("gof").

    Returns dict[mode] -> (r2 par spectre, nb pics par spectre).
    """
    n = flat_psds.shape[0]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in ("fixed", "knee"):
        fg = SpectralGroupModel(aperiodic_mode=mode, max_n_peaks=max_peaks, verbose=False)
        fg.fit(freqs, flat_psds, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)
        r2  = np.array(fg.get_params("r_squared"))   # (n_spectra,) — API specparam 2.0
        npk = count_peaks_per_spectrum(fg, n)
        out[mode] = (r2, npk)
    return out


def sample_epochs(data: np.ndarray, max_epochs: int, rng: np.random.RandomState) -> np.ndarray:
    """(n_epochs, 19, 7500) -> sous-échantillon (k, 19, 7500), k <= max_epochs."""
    n = data.shape[0]
    if n <= max_epochs:
        return data
    idx = rng.choice(n, size=max_epochs, replace=False)
    return data[idx]


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args     = parse_args()
    subjects = args.subjects if args.subjects is not None else SUBJECT_IDS
    rng      = np.random.RandomState(args.seed)

    # accumulation par stade : on empile r2/npeaks de tous les epochs échantillonnés
    acc: dict[str, dict[str, list[np.ndarray]]] = {
        s: {"r2_fixed": [], "r2_knee": [], "np_fixed": [], "np_knee": []}
        for s in ATOMIC_STAGES
    }

    for sub_id in subjects:
        if not _vhdr(args.deriv_path, sub_id).exists():
            print(f"sub-{sub_id}: derivative absent, skip")
            continue
        print(f"sub-{sub_id}: chargement epochs...")
        try:
            epochs = load_epochs_by_atomic_stage(args.deriv_path, sub_id)
        except Exception as e:
            print(f"  sub-{sub_id}: erreur chargement ({e}), skip")
            continue

        for stage, data in epochs.items():
            sample   = sample_epochs(data, args.max_epochs, rng)
            psds, freqs = compute_psd_spectrum(sample)        # (k, 19, n_freqs)
            n_freqs  = psds.shape[-1]
            flat     = psds.reshape(-1, n_freqs)              # (k*19, n_freqs)

            res = fit_both_modes(flat, freqs, args.max_peaks)
            acc[stage]["r2_fixed"].append(res["fixed"][0])
            acc[stage]["r2_knee"].append(res["knee"][0])
            acc[stage]["np_fixed"].append(res["fixed"][1])
            acc[stage]["np_knee"].append(res["knee"][1])
            print(f"  {stage}: {sample.shape[0]} epochs × 19 = {flat.shape[0]} spectres")

    # ─── tableau de synthèse par stade ────────────────────────────────────────
    rows = []
    for stage in ATOMIC_STAGES:
        a = acc[stage]
        if not a["r2_fixed"]:
            continue
        r2f = np.concatenate(a["r2_fixed"])
        r2k = np.concatenate(a["r2_knee"])
        npf = np.concatenate(a["np_fixed"])
        npk = np.concatenate(a["np_knee"])
        rows.append(dict(
            stage        = stage,
            n_spectra    = len(r2f),
            R2_fixed     = round(float(r2f.mean()), 4),
            R2_knee      = round(float(r2k.mean()), 4),
            delta_R2     = round(float(r2k.mean() - r2f.mean()), 4),
            npeaks_fixed = round(float(npf.mean()), 2),
            npeaks_knee  = round(float(npk.mean()), 2),
            delta_npeaks = round(float(npf.mean() - npk.mean()), 2),
        ))

    if not rows:
        print("\nAucun spectre traité — vérifier --deriv-path et --subjects.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("COMPARAISON FIXED vs KNEE (moyennes par stade)")
    print("=" * 70)
    print(df.to_string(index=False))

    # ─── verdict global ───────────────────────────────────────────────────────
    mean_dR2 = df["delta_R2"].mean()
    mean_dNP = df["delta_npeaks"].mean()
    print("\n" + "-" * 70)
    print(f"delta_R2 moyen     = {mean_dR2:+.4f}  (knee - fixed ; >0 = knee meilleur)")
    print(f"delta_npeaks moyen = {mean_dNP:+.2f}   (fixed - knee ; >0 = fixed surajoute)")
    print("-" * 70)
    if mean_dR2 > 0.02 or mean_dNP > 0.8:
        print("VERDICT : signe d'un coude réel -> passer feat_extract en aperiodic_mode='knee'.")
    elif mean_dR2 < 0.01 and abs(mean_dNP) < 0.5:
        print("VERDICT : spectre ~linéaire sur 1-45Hz -> garder 'fixed' (à documenter).")
    else:
        print("VERDICT : zone grise -> inspecter quelques fits superposés avant de trancher.")
