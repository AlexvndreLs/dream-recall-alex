#!/usr/bin/env python3
"""
plot_slide_features.py

Genere 6 figures pedagogiques (diapo "New features" et rapport) a partir
d'UN SEUL sujet, UNE SEULE epoch de 3 s, UN SEUL canal.

Figures produites (dans --outdir) :
  1. fooof_decomposition.png      -> PSD log-log + fit aperiodique + oscillations
                                     (illustre exposant aperiodique + oscillatory power ratio)
  2. permutation_entropy.png      -> motifs ordinaux (order=3) sur un bout de signal
  3. higuchi_fd.png               -> droite log(L_k) vs log(1/k), pente = HFD (kmax=10)
  4. spectral_entropy.png         -> PSD normalisee en distribution de proba + entropie
  5. specparam_equation.png       -> equation du modele specparam terme a terme
                                     (fond A(f) en haut, residu logsub en bas)
  6. lzc_illustration.png         -> complexite de Lempel-Ziv : binarisation
                                     mediane + parsing LZ76

Echecs explicites : si le .fif est introuvable ou le canal absent, on plante
avec un message clair (pas de fallback silencieux).

Usage :
  python3 plot_slide_features.py \
      --fif /scratch/alouis/dream_bids/derivatives/ica/sub-01_task-sleep_ica.fif \
      --channel Cz \
      --outdir ./plot_slide \
      --tmin 300.0
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def die(msg):
    print(f"[ERREUR] {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fif", required=True,
                   help="Chemin vers le signal source : .vhdr (BIDS clean) ou .fif")
    p.add_argument("--channel", default=None,
                   help="Nom du canal. Si absent, prend le premier canal EEG.")
    p.add_argument("--tmin", type=float, default=300.0,
                   help="Debut de la fenetre (en s), defaut 300 s")
    p.add_argument("--dur", type=float, default=3.0, help="Duree de l'epoch (s), defaut 3.0")
    p.add_argument("--outdir", default="./plot_slide", help="Dossier de sortie")
    p.add_argument("--pipeline-dur", type=float, default=None,
                   help="Duree de fenetre (s) utilisee uniquement par les figures "
                        "specparam_equation et lzc_illustration. Mettre 30 pour "
                        "coller aux epochs du pipeline et lisser le spectre. Par "
                        "defaut reutilise --dur, donc les 4 figures historiques "
                        "restent strictement inchangees.")
    p.add_argument("--fmin", type=float, default=1.0, help="Freq min pour PSD/FOOOF")
    p.add_argument("--fmax", type=float, default=45.0, help="Freq max pour PSD/FOOOF")
    p.add_argument("--events", default=None,
                   help="Chemin vers le _events.tsv BIDS. Si fourni, les figures Higuchi et "
                        "spectral entropy comparent SWS (S4) vs REM (R).")
    p.add_argument("--sws-label", default="Sleep stage S4",
                   help="Libelle du stage SWS dans events.tsv (defaut 'Sleep stage S4')")
    p.add_argument("--rem-label", default="Sleep stage R",
                   help="Libelle du stage REM dans events.tsv (defaut 'Sleep stage R')")
    p.add_argument("--sws-tmin", type=float, default=None,
                   help="tmin de l'epoch 1 pour Higuchi (sinon 1er bloc continu).")
    p.add_argument("--rem-tmin", type=float, default=None,
                   help="tmin de l'epoch 2 pour Higuchi (sinon 1er bloc continu).")
    p.add_argument("--se-sws-tmin", type=float, default=None,
                   help="tmin de l'epoch 1 pour spectral entropy. Si absent, reutilise --sws-tmin.")
    p.add_argument("--se-rem-tmin", type=float, default=None,
                   help="tmin de l'epoch 2 pour spectral entropy. Si absent, reutilise --rem-tmin.")
    return p.parse_args()


def find_stage_window(events_tsv, stage_label, dur, sfreq):
    """Trouve le debut (en s) d'un bloc continu de >= dur secondes du stage donne.
    Les events BIDS sont des marqueurs de 1 s ; on cherche une suite continue."""
    import csv
    onsets = []
    with open(events_tsv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # nettoie d'eventuels espaces/BOM residuels sur les noms de colonnes
        reader.fieldnames = [fn.strip().lstrip("\ufeff") for fn in reader.fieldnames]
        for row in reader:
            tt = row.get("trial_type", "")
            if stage_label in tt:
                onsets.append(float(row["onset"]))
    if not onsets:
        die(f"Aucun event contenant '{stage_label}' dans {events_tsv}")
    onsets = sorted(set(onsets))
    # cherche une plage continue (pas de trou > 1 s) d'au moins dur secondes
    start = onsets[0]
    prev = onsets[0]
    for o in onsets[1:]:
        if o - prev > 1.5:  # rupture de continuite
            if prev - start + 1 >= dur:
                return start
            start = o
        prev = o
    if prev - start + 1 >= dur:
        return start
    die(f"Pas de bloc continu de {dur}s pour '{stage_label}' (max trouve plus court).")


def open_raw(fif_path, channel):
    """Ouvre le raw SANS preload, resout le canal, renvoie (raw, channel, sfreq)."""
    import mne
    if not os.path.isfile(fif_path):
        die(f"Fichier introuvable : {fif_path}")
    if fif_path.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(fif_path, preload=False, verbose="ERROR")
    elif fif_path.endswith(".fif"):
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    else:
        die(f"Extension non geree : {fif_path} (attendu .vhdr ou .fif)")
    sfreq = raw.info["sfreq"]
    if channel is None:
        picks = mne.pick_types(raw.info, eeg=True)
        if len(picks) == 0:
            die("Aucun canal EEG trouve dans le fichier.")
        channel = raw.ch_names[picks[0]]
        print(f"[info] Canal non specifie, utilisation de : {channel}")
    if channel not in raw.ch_names:
        die(f"Canal '{channel}' absent. Canaux dispo : {raw.ch_names}")
    return raw, channel, sfreq


def extract_window(raw, channel, sfreq, tmin, dur, label=""):
    """Extrait une fenetre de dur s a partir de tmin, pour 1 canal, sans OOM."""
    tmax = tmin + dur
    if tmax > raw.times[-1]:
        die(f"tmin+dur={tmax:.1f}s depasse la duree du signal ({raw.times[-1]:.1f}s).")
    seg = raw.copy().pick([channel]).crop(tmin=tmin, tmax=tmax - 1.0 / sfreq)
    seg.load_data(verbose="ERROR")
    sig = seg.get_data()[0]
    tag = f" [{label}]" if label else ""
    print(f"[info] Epoch{tag} : {channel}, {len(sig)} ech, [{tmin:.1f}-{tmax:.1f}]s")
    return sig


def compute_psd(sig, sfreq, fmin, fmax):
    from scipy.signal import welch
    nper = min(len(sig), int(sfreq * 2))  # fenetres de 2 s max
    freqs, psd = welch(sig, fs=sfreq, nperseg=nper)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


# ---------------------------------------------------------------------------
# 1. FOOOF / specparam
# ---------------------------------------------------------------------------
def plot_fooof(sig, sfreq, fmin, fmax, out, channel, use_specparam=True):
    freqs, psd = compute_psd(sig, sfreq, fmin, fmax)
    lf = np.log10(freqs)
    lp = np.log10(psd)

    # Composante aperiodique de reference : regression lineaire en log-log (fallback).
    slope, intercept = np.polyfit(lf, lp, 1)
    ap_log = intercept + slope * lf
    exponent = -slope
    n_peaks = None

    # specparam : fit du modele complet. On recupere l'aperiodique via get_model
    # pour que la soustraction PSD - aperiodique corresponde exactement au modele.
    if use_specparam:
        try:
            from specparam import SpectralModel
            sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                               aperiodic_mode="fixed", verbose=False)
            sm.fit(freqs, psd)
            exponent = sm.get_params("aperiodic", "exponent")
            offset = sm.get_params("aperiodic", "offset")
            ap_log = offset - exponent * np.log10(freqs)  # aperiodique en log10, mode fixed
            n_peaks = np.atleast_2d(sm.get_params("peak")).shape[0]
            print(f"[info] specparam OK, exposant={exponent:.2f}, {n_peaks} pic(s)")
        except Exception as e:
            print(f"[warn] specparam indisponible ({e}), fit log-log manuel.")

    ap_fit = 10 ** ap_log
    # Composante oscillatoire = PSD (log) moins aperiodique (log). C'est LA soustraction,
    # exactement le psd_osc du pipeline. Positive = puissance au-dessus du 1/f.
    osc_log = lp - ap_log

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7),
                                   gridspec_kw={"height_ratios": [2, 1]})
    # panneau haut : PSD + aperiodique
    ax1.loglog(freqs, psd, color="black", lw=2, label="Observed PSD")
    ax1.loglog(freqs, ap_fit, color="tab:red", ls="--", lw=2,
               label=f"Aperiodic component (1/f, exponent={exponent:.2f})")
    ax1.fill_between(freqs, ap_fit, psd, where=(psd > ap_fit),
                     color="tab:blue", alpha=0.18, label="Oscillatory power (above 1/f)")
    ax1.set_ylabel("Power (log)")
    ttl = f"FOOOF: PSD = aperiodic (1/f) + oscillations ({channel})"
    ax1.set_title(ttl)
    ax1.legend(fontsize=8.5, loc="lower left")
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # panneau bas : la soustraction elle-meme, composante oscillatoire isolee
    ax2.axhline(0, color="tab:red", ls="--", lw=1.5)
    ax2.fill_between(freqs, 0, osc_log, where=(osc_log > 0),
                     color="tab:blue", alpha=0.5)
    ax2.plot(freqs, osc_log, color="tab:blue", lw=1.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD - aperiodic\n(log, = psd_osc)")
    ax2.set_title("Isolated oscillatory component (the subtraction)", fontsize=10)
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# 2. Permutation entropy (order=3) : motifs ordinaux
# ---------------------------------------------------------------------------
def plot_permutation_entropy(sig, sfreq, out, channel):
    from itertools import permutations
    order = 3
    # sous-echantillonne pour un schema lisible : ~40 points
    step = max(1, len(sig) // 40)
    s = sig[::step][:40]
    t = np.arange(len(s))

    # calcule la distribution des motifs ordinaux
    patterns = list(permutations(range(order)))
    counts = {p: 0 for p in patterns}
    for i in range(len(s) - order + 1):
        window = s[i:i + order]
        rank = tuple(np.argsort(window))
        counts[rank] += 1
    total = sum(counts.values())
    probs = np.array([counts[p] / total for p in patterns])
    pe = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    pe_norm = pe / np.log2(len(patterns))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    # panneau gauche : signal + fenetre glissante mise en avant
    ax1.plot(t, s, "-o", color="black", ms=4, lw=1)
    hi = 5  # fenetre exemple
    ax1.plot(t[hi:hi + order], s[hi:hi + order], "-o", color="tab:red", ms=8, lw=2.5)
    win = s[hi:hi + order]
    rank = tuple(int(x) for x in np.argsort(win))
    ax1.set_title(f"Signal split into {order}-point windows\n"
                  f"example ordinal pattern: {rank}")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, ls=":", alpha=0.4)

    # panneau droit : histogramme des 6 motifs
    labels = ["".join(str(x) for x in p) for p in patterns]
    ax2.bar(range(len(patterns)), probs, color="tab:blue", alpha=0.8)
    ax2.set_xticks(range(len(patterns)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel(f"Ordinal patterns (3! = {len(patterns)})")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Permutation entropy (normalized) = {pe_norm:.2f}")
    ax2.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle(f"Permutation entropy, order=3 ({channel})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# 3. Higuchi fractal dimension (kmax=10)
# ---------------------------------------------------------------------------
def _higuchi_fd(sig, kmax=10):
    """Retourne (HFD, x=log(1/k), y=log(L_k)) pour un signal."""
    N = len(sig)
    Lk = []
    ks = np.arange(1, kmax + 1)
    for k in ks:
        Lm = []
        for m in range(k):
            idx = np.arange(1, int((N - m) / k))
            if len(idx) == 0:
                continue
            Lmk = np.sum(np.abs(sig[m + idx * k] - sig[m + (idx - 1) * k]))
            norm = (N - 1) / (len(idx) * k)
            Lm.append(Lmk * norm / k)
        Lk.append(np.mean(Lm))
    Lk = np.array(Lk)
    x = np.log(1.0 / ks)
    y = np.log(Lk)
    slope, _ = np.polyfit(x, y, 1)
    return slope, x, y


def plot_higuchi_compare(sig_sws, sig_rem, sfreq, out, channel, kmax=10):
    hfd_sws, _, _ = _higuchi_fd(sig_sws, kmax)
    hfd_rem, _, _ = _higuchi_fd(sig_rem, kmax)
    # un extrait court pour la lisibilite (1 s)
    n = int(sfreq * 1)
    t = np.arange(n) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, sig_sws[:n] * 1e6, color="tab:blue", lw=1)
    axes[0].set_title(f"Epoch 1, HFD = {hfd_sws:.2f}", fontsize=12)
    axes[0].set_ylabel("uV")
    axes[1].plot(t, sig_rem[:n] * 1e6, color="tab:red", lw=1)
    axes[1].set_title(f"Epoch 2, HFD = {hfd_rem:.2f}", fontsize=12)
    axes[1].set_ylabel("uV")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"Higuchi fractal dimension ({channel})", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_higuchi(sig, out, channel, kmax=10):
    """Version simple (une epoch) conservee en fallback si pas d'events."""
    slope, x, y = _higuchi_fd(sig, kmax)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x, y, "o", color="black", ms=8, label="measured log L(k)")
    ax.plot(x, np.polyval(np.polyfit(x, y, 1), x), "-", color="tab:red", lw=2,
            label=f"slope = HFD = {slope:.2f}")
    ax.set_xlabel("log(1/k)"); ax.set_ylabel("log(L(k))")
    ax.set_title(f"Higuchi fractal dimension, kmax={kmax} ({channel})")
    ax.legend(fontsize=10); ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# 4. Spectral entropy
# ---------------------------------------------------------------------------
def _spectral_entropy(sig, sfreq, fmin, fmax):
    freqs, psd = compute_psd(sig, sfreq, fmin, fmax)
    p = psd / np.sum(psd)
    se = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return freqs, p, se / np.log2(len(p))


def plot_spectral_entropy_compare(sig_sws, sig_rem, sfreq, fmin, fmax, out, channel):
    f_sws, p_sws, se_sws = _spectral_entropy(sig_sws, sfreq, fmin, fmax)
    f_rem, p_rem, se_rem = _spectral_entropy(sig_rem, sfreq, fmin, fmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ax1.fill_between(f_sws, p_sws, color="tab:blue", alpha=0.6, step="mid")
    ax1.set_title(f"Epoch 1, spectral entropy = {se_sws:.2f}", fontsize=11)
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Normalized P(f)")
    ax2.fill_between(f_rem, p_rem, color="tab:red", alpha=0.6, step="mid")
    ax2.set_title(f"Epoch 2, spectral entropy = {se_rem:.2f}", fontsize=11)
    ax2.set_xlabel("Frequency (Hz)")
    for ax in (ax1, ax2):
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"Spectral entropy ({channel})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_spectral_entropy(sig, sfreq, fmin, fmax, out, channel):
    """Version simple (une epoch) conservee en fallback."""
    freqs, p, se_norm = _spectral_entropy(sig, sfreq, fmin, fmax)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(freqs, p, color="tab:blue", alpha=0.6, step="mid")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Normalized P(f)")
    ax.set_title(f"Spectral entropy (normalized) = {se_norm:.2f} ({channel})")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# 5. Equation du modele specparam (ex-FOOOF)
# ---------------------------------------------------------------------------
def _fit_specparam(freqs, psd, max_n_peaks=6):
    """Ajuste specparam en mode 'fixed' et renvoie les composantes en log10.

    Retourne (offset, exponent, peaks, ap_log, peak_log, model_log) ou peaks est
    un tableau (n_peaks, 3) de colonnes (CF, PW, BW) : centre en Hz, hauteur
    au-dessus de l'aperiodique en log10, largeur a mi-hauteur en Hz.

    La composante aperiodique en mode 'fixed' est log10 A(f) = b - chi * log10(f).
    La composante periodique est recuperee via l'API du modele quand elle est
    disponible ; a defaut elle est reconstruite comme une somme de gaussiennes
    d'ecart-type BW/2, ce qui est la definition du modele.
    """
    from specparam import SpectralModel

    sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=max_n_peaks,
                       aperiodic_mode="fixed", verbose=False)
    sm.fit(freqs, psd)
    offset = float(sm.get_params("aperiodic", "offset"))
    exponent = float(sm.get_params("aperiodic", "exponent"))
    peaks = np.atleast_2d(sm.get_params("peak"))
    if peaks.size == 0:
        peaks = np.zeros((0, 3))

    ap_log = offset - exponent * np.log10(freqs)
    try:  # specparam 2.x : composante periodique exacte du modele ajuste
        peak_log = np.asarray(sm.results.model.get_component("peak", "log"))
    except Exception:
        peak_log = np.zeros_like(freqs)
        for cf, pw, bw in peaks:
            peak_log = peak_log + pw * np.exp(-(freqs - cf) ** 2 / (2 * (bw / 2.0) ** 2))
    return offset, exponent, peaks, ap_log, peak_log, ap_log + peak_log


def plot_specparam_equation(sig, sfreq, fmin, fmax, out, channel, max_n_peaks=6,
                            band=(11.0, 16.0), band_name="sigma"):
    """Illustration en deux panneaux du modele specparam.

    Haut : spectre observe et fond aperiodique ajuste.
    Bas  : la definition logsub, residu signe, et la moyenne sur une bande.
    Volontairement depouille : deux courbes en haut, deux en bas.
    """
    freqs, psd = compute_psd(sig, sfreq, fmin, fmax)
    lp = np.log10(psd)
    offset, exponent, peaks, ap_log, peak_log, model_log = _fit_specparam(
        freqs, psd, max_n_peaks)
    flat = lp - ap_log

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1]})

    # --- haut : signal observe et fond aperiodique
    ax1.plot(freqs, lp, color="black", lw=1.8, label="Observed spectrum")
    ax1.plot(freqs, ap_log, color="tab:red", lw=2.2, ls="--",
             label=f"Aperiodic fit A(f): $b$ = {offset:.2f}, "
                   f"$\\chi$ = {exponent:.2f}")
    ax1.set_xscale("log")
    ax1.set_ylabel("log10 power")
    ax1.set_title(r"$\log_{10} P(f) = \log_{10} A(f) + \sum_{n} G_n(f)$"
                  + f"      ({channel})", fontsize=12.5)
    ax1.legend(fontsize=9.5, loc="lower left")
    ax1.grid(True, which="both", ls=":", alpha=0.35)

    # --- bas : la definition logsub, residu signe
    ax2.axhline(0, color="tab:red", ls="--", lw=1.3)
    ax2.plot(freqs, flat, color="black", lw=1.4)
    # une bande, pour montrer d'ou sort la valeur scalaire
    b0, b1 = band
    ax2.axvspan(b0, b1, color="gray", alpha=0.12)
    inb = (freqs >= b0) & (freqs <= b1)
    if inb.any():
        ax2.annotate(f"{band_name} band: logsub = {flat[inb].mean():.2f}",
                     xy=((b0 + b1) / 2, flat[inb].mean()),
                     xytext=(0.02, 0.86), textcoords="axes fraction", fontsize=9.5,
                     arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.1))
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"$\log_{10} P(f) - \log_{10} A(f)$")
    ax2.set_title(r"$\mathtt{logsub} = \frac{1}{|B|} \sum_{f \in B} "
                  r"\left[ \log_{10} P(f) - \log_{10} A(f) \right]$", fontsize=12)
    ax2.grid(True, which="both", ls=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}  (offset={offset:.2f}, exponent={exponent:.2f}, "
          f"{len(peaks)} peak(s))")


# ---------------------------------------------------------------------------
# 6. Lempel-Ziv complexity (LZ76)
# ---------------------------------------------------------------------------
def _binarize_median(sig):
    """Binarisation par rapport a la mediane : 1 au-dessus, 0 en dessous."""
    return (sig > np.median(sig)).astype(int)


def _lz76_phrases(bits):
    """Parsing LZ76 exhaustif. Renvoie la liste des motifs distincts.

    Chaque nouveau motif est le plus court prefixe de la suite restante qui
    n'apparait pas deja comme sous-chaine de l'historique. Le nombre de motifs
    est la complexite brute c(n).
    """
    s = "".join(str(int(b)) for b in bits)
    n, i, phrases = len(s), 0, []
    while i < n:
        L = 1
        while i + L <= n and s[i:i + L] in s[:i + L - 1]:
            L += 1
        phrases.append(s[i:i + L])
        i += L
    return phrases


def _lzc(bits):
    """(c_raw, lzc_normalisee). Normalisation c(n) * log2(n) / n, soit la
    complexite d'une sequence aleatoire de meme longueur (convention antropy,
    normalize=True)."""
    n = len(bits)
    c_raw = len(_lz76_phrases(bits))
    return c_raw, c_raw * np.log2(n) / n


def _lzc_panel(ax_sig, ax_bin, sig, sfreq, color, label, n_show=64, excerpt_s=4.0):
    """Trace un couple (signal + chaine binaire parsee) pour une epoch.

    La valeur LZC du titre porte sur l'epoch entiere. L'extrait dessine est une
    fenetre continue de excerpt_s secondes, reduite a n_show points par moyenne
    par blocs (le moyennage fait office de passe-bas, il n'y a pas de repliement
    comme avec un simple sous-echantillonnage).
    """
    c_raw, lzc = _lzc(_binarize_median(sig))

    win = int(min(len(sig), excerpt_s * sfreq))
    start = (len(sig) - win) // 2
    blk = max(1, win // n_show)
    npts = win // blk
    ex = sig[start:start + npts * blk].reshape(npts, blk).mean(axis=1)
    bits_ex = _binarize_median(ex)
    med = np.median(ex)
    t = np.arange(npts) * blk / sfreq

    ax_sig.plot(t, ex * 1e6, "-o", color=color, ms=3, lw=1.2)
    ax_sig.axhline(med * 1e6, color="black", ls="--", lw=1.2, label="median")
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("uV")
    ax_sig.set_title(f"{label}: LZC = {lzc:.2f}  (c(n) = {c_raw} patterns "
                     f"over the whole epoch)", fontsize=11)
    ax_sig.legend(fontsize=8, loc="upper right")
    ax_sig.grid(True, ls=":", alpha=0.4)

    # chaine binaire : un carre par echantillon, separateurs entre motifs
    phrases = _lz76_phrases(bits_ex)
    for k, b in enumerate(bits_ex):
        ax_bin.add_patch(plt.Rectangle((k, 0), 1, 1,
                                       facecolor=color if b else "gainsboro",
                                       edgecolor="white", lw=0.5,
                                       alpha=0.9 if b else 1.0))
    pos = 0
    for ph in phrases:
        pos += len(ph)
        ax_bin.plot([pos, pos], [-0.18, 1.18], color="black", lw=1.8)
    ax_bin.set_xlim(0, npts)
    ax_bin.set_ylim(-0.45, 1.45)
    ax_bin.set_yticks([])
    ax_bin.set_xlabel("Binarized excerpt: colored = above median, gray = below")
    ax_bin.set_title(f"LZ76 parsing, one bar closes one distinct pattern: "
                     f"{len(phrases)} patterns here", fontsize=10)


def plot_lzc(sig_a, sig_b, sfreq, out, channel, n_show=64):
    """Illustration de la complexite de Lempel-Ziv sur une ou deux epochs.

    Colonne de gauche : signal et seuil de binarisation (la mediane).
    Colonne de droite : chaine binaire, decoupee par le parsing LZ76 ; chaque
    barre verticale ferme un motif distinct.

    La valeur LZC affichee est calculee sur l'epoch entiere ; le parsing dessine
    porte sur un extrait sous-echantillonne, seul moyen de le rendre lisible.
    """
    rows = 2 if sig_b is not None else 1
    fig, axes = plt.subplots(rows, 2, figsize=(12.5, 3.3 * rows + 0.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    axes = np.atleast_2d(axes)

    _lzc_panel(axes[0, 0], axes[0, 1], sig_a, sfreq, "tab:blue", "Epoch 1", n_show)
    if sig_b is not None:
        _lzc_panel(axes[1, 0], axes[1, 1], sig_b, sfreq, "tab:red", "Epoch 2", n_show)

    fig.suptitle(f"Lempel-Ziv complexity ({channel}). "
                 "Normalization: c(n) log2(n) / n", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    raw, ch, sfreq = open_raw(args.fif, args.channel)

    if args.events:
        # Paire pour Higuchi (epochs 1 et 2, forcees ou 1er bloc continu).
        t1 = args.sws_tmin if args.sws_tmin is not None \
            else find_stage_window(args.events, args.sws_label, args.dur, sfreq)
        t2 = args.rem_tmin if args.rem_tmin is not None \
            else find_stage_window(args.events, args.rem_label, args.dur, sfreq)
        print(f"[info] Higuchi : epoch1 a {t1:.1f}s, epoch2 a {t2:.1f}s")
        sig1 = extract_window(raw, ch, sfreq, t1, args.dur, "HFD-1")
        sig2 = extract_window(raw, ch, sfreq, t2, args.dur, "HFD-2")

        # Paire pour spectral entropy : tmin dedies si fournis, sinon reutilise ceux de Higuchi.
        se_t1 = args.se_sws_tmin if args.se_sws_tmin is not None else t1
        se_t2 = args.se_rem_tmin if args.se_rem_tmin is not None else t2
        if (se_t1, se_t2) == (t1, t2):
            se_sig1, se_sig2 = sig1, sig2
        else:
            print(f"[info] Spectral entropy : epoch1 a {se_t1:.1f}s, epoch2 a {se_t2:.1f}s")
            se_sig1 = extract_window(raw, ch, sfreq, se_t1, args.dur, "SE-1")
            se_sig2 = extract_window(raw, ch, sfreq, se_t2, args.dur, "SE-2")

        # FOOOF et permutation entropy : sur l'epoch 1 de Higuchi.
        plot_fooof(sig1, sfreq, args.fmin, args.fmax,
                   os.path.join(args.outdir, "fooof_decomposition.png"), ch)
        plot_permutation_entropy(sig1, sfreq,
                   os.path.join(args.outdir, "permutation_entropy.png"), ch)
        # Higuchi : paire dediee HFD.
        plot_higuchi_compare(sig1, sig2, sfreq,
                   os.path.join(args.outdir, "higuchi_fd.png"), ch, kmax=10)
        # Spectral entropy : paire dediee SE.
        plot_spectral_entropy_compare(se_sig1, se_sig2, sfreq, args.fmin, args.fmax,
                   os.path.join(args.outdir, "spectral_entropy.png"), ch)
        # Fenetre dediee aux deux nouvelles figures (--pipeline-dur, sinon --dur).
        long_dur = args.pipeline_dur if args.pipeline_dur is not None else args.dur
        if long_dur != args.dur:
            sig1_long = extract_window(raw, ch, sfreq, t1, long_dur, "LONG-1")
            sig2_long = extract_window(raw, ch, sfreq, t2, long_dur, "LONG-2")
        else:
            sig1_long, sig2_long = sig1, sig2
        # Equation specparam : sur l'epoch 1 de Higuchi, comme FOOOF.
        plot_specparam_equation(sig1_long, sfreq, args.fmin, args.fmax,
                   os.path.join(args.outdir, "specparam_equation.png"), ch)
        # LZC : paire d'epochs, meme logique comparative que Higuchi.
        plot_lzc(sig1_long, sig2_long, sfreq,
                   os.path.join(args.outdir, "lzc_illustration.png"), ch)
    else:
        # Mode simple : une seule epoch a tmin.
        sig = extract_window(raw, ch, sfreq, args.tmin, args.dur)
        plot_fooof(sig, sfreq, args.fmin, args.fmax,
                   os.path.join(args.outdir, "fooof_decomposition.png"), ch)
        plot_permutation_entropy(sig, sfreq,
                   os.path.join(args.outdir, "permutation_entropy.png"), ch)
        plot_higuchi(sig,
                   os.path.join(args.outdir, "higuchi_fd.png"), ch, kmax=10)
        plot_spectral_entropy(sig, sfreq, args.fmin, args.fmax,
                   os.path.join(args.outdir, "spectral_entropy.png"), ch)
        long_dur = args.pipeline_dur if args.pipeline_dur is not None else args.dur
        sig_long = sig if long_dur == args.dur else extract_window(
            raw, ch, sfreq, args.tmin, long_dur, "LONG")
        plot_specparam_equation(sig_long, sfreq, args.fmin, args.fmax,
                   os.path.join(args.outdir, "specparam_equation.png"), ch)
        plot_lzc(sig_long, None, sfreq,
                   os.path.join(args.outdir, "lzc_illustration.png"), ch)

    print(f"\n[fini] 6 figures dans {args.outdir}/")


if __name__ == "__main__":
    main()