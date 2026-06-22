import numpy as np, mne, mne_bids
from pathlib import Path
mne.set_log_level('ERROR')
from config_v3 import HP_FREQ_ICA

BIDS = Path('/home/alouis/scratch/dream_bids')
DERIV = Path('/home/alouis/scratch/dream_bids/derivatives')
sub = '05'

raw = mne_bids.read_raw_bids(mne_bids.BIDSPath(
    subject=sub, task='sleep', root=BIDS, datatype='eeg'), verbose=False)
raw.load_data()
raw_for_ica = raw.copy()
raw_for_ica.filter(l_freq=HP_FREQ_ICA, h_freq=None, verbose=False)
ica = mne.preprocessing.read_ica(DERIV / 'ica' / f'sub-{sub}_task-sleep_ica.fif')

print("=== ETAT ===")
print("n_components_ :", ica.n_components_)
print("ica.ch_names (%d) :" % len(ica.ch_names), ica.ch_names)
print("raw ch types :", {c: t for c, t in zip(raw_for_ica.ch_names,
      raw_for_ica.get_channel_types())})

# --- reproduit la logique interne de find_bads_muscle pour voir les tailles ---
print("\n=== TAILLES DES 3 CRITERES (raw complet) ===")
try:
    sources = ica.get_sources(raw_for_ica)
    comp = ica.get_components()
    print("sources n_ch :", len(sources.ch_names),
          "| sources types uniques :", set(sources.get_channel_types()))
    spec = sources.compute_psd(fmin=7, fmax=45, picks="misc")
    psds, freqs = spec.get_data(return_freqs=True)
    print("psds shape (picks=misc) :", psds.shape, "  <-- nb de 'misc' captes")
    print("components shape :", comp.shape, "  (n_ch, n_comp)")
    print(">>> slope_score aura len =", psds.shape[0],
          "| focus/smoothness auront len =", comp.shape[1])
    if psds.shape[0] != comp.shape[1]:
        print(">>> MISMATCH CONFIRME : c'est la cause du crash np.prod")
    else:
        print(">>> pas de mismatch ici")
except Exception as e:
    print("erreur en reproduisant :", type(e).__name__, str(e)[:120])

print("\n=== VARIANTES DE FIX ===")
for name, builder in [
    ("A raw complet",      lambda: raw_for_ica),
    ("B pick(ica.ch_names)", lambda: raw_for_ica.copy().pick(ica.ch_names)),
    ("C pick('eeg')",      lambda: raw_for_ica.copy().pick('eeg')),
]:
    try:
        _, s = ica.find_bads_muscle(builder(), threshold=0, verbose=False)
        print(f"  {name}: OK shape {np.array(s).shape}")
    except Exception as e:
        print(f"  {name}: CRASH {type(e).__name__} {str(e)[:80]}")
