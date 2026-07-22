#!/usr/bin/env python3
"""
find_contrast_epochs.py

Scanne les blocs continus de deux stages (SWS et REM) et calcule la HFD de chacun,
puis renvoie la paire (SWS, REM) avec le PLUS GRAND ecart de HFD. Sert a choisir
deux epochs visuellement contrastees pour la diapo Higuchi.

Affiche les tmin a passer ensuite a plot_slide_features.py.

Usage :
  python3 find_contrast_epochs.py \
    --fif   .../sub-01_task-sleep_proc-clean_eeg.vhdr \
    --events .../sub-01_task-sleep_proc-clean_events.tsv \
    --channel Cz --dur 30.0
"""
import argparse, csv, sys, os
import numpy as np


def die(m):
    print(f"[ERREUR] {m}", file=sys.stderr); sys.exit(1)


def higuchi_fd(sig, kmax=10):
    N = len(sig); Lk = []; ks = np.arange(1, kmax + 1)
    for k in ks:
        Lm = []
        for m in range(k):
            idx = np.arange(1, int((N - m) / k))
            if len(idx) == 0:
                continue
            Lmk = np.sum(np.abs(sig[m + idx * k] - sig[m + (idx - 1) * k]))
            Lm.append(Lmk * (N - 1) / (len(idx) * k) / k)
        Lk.append(np.mean(Lm))
    x = np.log(1.0 / ks); y = np.log(np.array(Lk))
    return np.polyfit(x, y, 1)[0]


def spectral_entropy(sig, sfreq, fmin=1.0, fmax=45.0):
    from scipy.signal import welch
    nper = min(len(sig), int(sfreq * 2))
    freqs, psd = welch(sig, fs=sfreq, nperseg=nper)
    mask = (freqs >= fmin) & (freqs <= fmax)
    p = psd[mask]; p = p / np.sum(p)
    se = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return se / np.log2(len(p))


def continuous_blocks(events_tsv, stage_label, dur):
    """Renvoie la liste des tmin de tous les blocs continus >= dur pour ce stage."""
    onsets = []
    with open(events_tsv, encoding="utf-8-sig") as f:
        r = csv.DictReader(f, delimiter="\t")
        r.fieldnames = [fn.strip().lstrip("\ufeff") for fn in r.fieldnames]
        for row in r:
            if stage_label in row.get("trial_type", ""):
                onsets.append(float(row["onset"]))
    if not onsets:
        die(f"Aucun event '{stage_label}'")
    onsets = sorted(set(onsets))
    blocks = []
    start = prev = onsets[0]
    for o in onsets[1:]:
        if o - prev > 1.5:
            if prev - start + 1 >= dur:
                blocks.append(start)
            start = o
        prev = o
    if prev - start + 1 >= dur:
        blocks.append(start)
    return blocks


def scan_stage(raw, ch, sfreq, blocks, dur, stage_name, max_candidates=12):
    """Calcule HFD et spectral entropy sur (au plus) max_candidates blocs.
    Renvoie [(tmin, hfd, se)]."""
    if not blocks:
        die(f"Pas de bloc continu de {dur}s pour {stage_name}")
    if len(blocks) > max_candidates:
        idx = np.linspace(0, len(blocks) - 1, max_candidates).astype(int)
        blocks = [blocks[i] for i in idx]
    out = []
    for tmin in blocks:
        seg = raw.copy().pick([ch]).crop(tmin=tmin, tmax=tmin + dur - 1.0 / sfreq)
        seg.load_data(verbose="ERROR")
        sig = seg.get_data()[0]
        hfd = higuchi_fd(sig)
        se = spectral_entropy(sig, sfreq)
        out.append((tmin, hfd, se))
        print(f"  [{stage_name}] tmin={tmin:7.1f}s  HFD={hfd:.3f}  SE={se:.3f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fif", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--channel", default="Cz")
    p.add_argument("--dur", type=float, default=30.0)
    p.add_argument("--sws-label", default="Sleep stage S4")
    p.add_argument("--rem-label", default="Sleep stage R")
    args = p.parse_args()

    import mne
    if not os.path.isfile(args.fif):
        die(f"Introuvable : {args.fif}")
    raw = mne.io.read_raw_brainvision(args.fif, preload=False, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    if args.channel not in raw.ch_names:
        die(f"Canal '{args.channel}' absent. Dispo : {raw.ch_names}")

    print(f"Scan SWS ({args.sws_label}) :")
    sws = scan_stage(raw, args.channel, sfreq,
                     continuous_blocks(args.events, args.sws_label, args.dur),
                     args.dur, "SWS")
    print(f"Scan REM ({args.rem_label}) :")
    rem = scan_stage(raw, args.channel, sfreq,
                     continuous_blocks(args.events, args.rem_label, args.dur),
                     args.dur, "REM")

    # on fusionne tous les candidats des deux stages : on cherche juste 2 epochs
    # avec des valeurs bien differentes, peu importe le stage.
    allc = sws + rem  # [(tmin, hfd, se), ...]

    def best_pair(metric_idx, name):
        lo = min(allc, key=lambda z: z[metric_idx])
        hi = max(allc, key=lambda z: z[metric_idx])
        print("\n" + "=" * 60)
        print(f"MEILLEUR CONTRASTE {name} : ecart = {hi[metric_idx] - lo[metric_idx]:.3f}")
        print(f"  Epoch 1 (basse) : tmin={lo[0]:.1f}s  {name}={lo[metric_idx]:.3f}")
        print(f"  Epoch 2 (haute) : tmin={hi[0]:.1f}s  {name}={hi[metric_idx]:.3f}")
        print("=" * 60)
        return lo[0], hi[0]

    ts_h, tr_h = best_pair(1, "HFD")
    ts_s, tr_s = best_pair(2, "SE")

    print("\nPour la figure Higuchi :")
    print(f"  --sws-tmin {ts_h:.1f} --rem-tmin {tr_h:.1f}")
    print("Pour la figure spectral entropy (relancer separement avec ces tmin) :")
    print(f"  --sws-tmin {ts_s:.1f} --rem-tmin {tr_s:.1f}")


if __name__ == "__main__":
    main()