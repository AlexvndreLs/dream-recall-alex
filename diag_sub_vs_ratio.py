import numpy as np, collections
from pathlib import Path
from config_v3 import FREQ_DICT, FOOOF_FREQ_RANGE
from feat_extract_umap_fooof_v4 import (
    load_epochs_by_atomic_stage, compute_psd_spectrum, band_power)
from specparam import SpectralGroupModel

DERIV = Path("/home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica")
SUBS   = ["01", "02", "23", "24"]
STAGES = ["S2", "S3", "REM"]
NEP    = 15
K      = 1e12

rows = []
for sub in SUBS:
    try:
        at = load_epochs_by_atomic_stage(DERIV, sub)
    except Exception as e:
        print(f"sub-{sub} : echec chargement ({e}), skip", flush=True)
        continue
    for st in STAGES:
        if st not in at:
            print(f"sub-{sub} {st} : absent, skip", flush=True)
            continue
        data = at[st][:NEP]
        psds, freqs = compute_psd_spectrum(data)
        ne, nc, nf = psds.shape
        P = psds.reshape(-1, nf)

        fg = SpectralGroupModel(aperiodic_mode="fixed", verbose=False)
        fg.fit(freqs, P, freq_range=FOOOF_FREQ_RANGE, n_jobs=1)
        ap = fg.get_params("aperiodic")
        A = 10 ** (ap[:, 0:1] - ap[:, 1:2] * np.log10(freqs)[None, :])

        sub64 = P - A
        subhi = P.astype(np.longdouble) - A.astype(np.longdouble)
        subsc = ((P * K) - (A * K)) / K
        ratio = P / A

        ref = np.asarray(subhi, dtype=np.float64)
        m = np.abs(ref) > 0
        e_raw = np.median(np.abs(sub64[m] - ref[m]) / np.abs(ref[m]))
        e_sc  = np.median(np.abs(subsc[m] - ref[m]) / np.abs(ref[m]))
        canc  = np.median(np.abs(P[m]) / np.abs(sub64[m]))
        print(f"sub-{sub} {st:4s} ne={ne:3d} relerr={e_raw:.3e} "
              f"relerr_scaled={e_sc:.3e} cancel={canc:.3e}", flush=True)

        for b, (f0, f1) in FREQ_DICT.items():
            sh = (ne, nc, nf)
            r_ = band_power(ratio.reshape(sh), freqs, f0, f1).ravel()
            s_ = band_power(sub64.reshape(sh), freqs, f0, f1).ravel()
            ok = np.isfinite(r_) & np.isfinite(s_)
            r = np.corrcoef(r_[ok], s_[ok])[0, 1] if ok.sum() > 2 else np.nan
            rows.append((st, b, r, np.median(np.abs(s_)), (s_ < 0).mean()))

print("\nstage band        r   |sub|_med  frac_neg", flush=True)
agg = collections.defaultdict(list)
for st, b, r, a, f in rows:
    agg[(st, b)].append((r, a, f))
for (st, b), v in sorted(agg.items()):
    r = np.nanmean([x[0] for x in v])
    a = np.nanmean([x[1] for x in v])
    f = np.nanmean([x[2] for x in v])
    print(f"{st:5s} {b:6s} {r:7.4f} {a:.3e} {f:7.3f}", flush=True)
