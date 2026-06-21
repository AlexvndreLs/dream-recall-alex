import mne, mne_bids

root = "/home/alouis/scratch/dream_bids"
sub = "26"

raw = mne_bids.read_raw_bids(mne_bids.BIDSPath(subject=sub, task="sleep",
    root=root, datatype="eeg"), verbose=False)
raw.load_data()
raw_for_ica = raw.copy()
raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False)

# voie horizontale comme dans le pipeline
raw_eog = mne.set_bipolar_reference(raw_for_ica, anode="EOG_L", cathode="EOG_R",
    ch_name="EOG_horiz", drop_refs=False, copy=True, verbose=False)
raw_eog.set_channel_types({"EOG_horiz": "eog"}, verbose=False)

ica = mne.preprocessing.read_ica(f"{root}/derivatives/ica/sub-{sub}_task-sleep_ica.fif")

print("=== z-score (actuel) ===")
for t in (2.0, 2.5, 3.0):
    idx, scores = ica.find_bads_eog(raw_eog, ch_name=["EOG_L","EOG_R","EOG_horiz"],
                                     threshold=t, measure="zscore", verbose=False)
    print(f"  zscore {t} -> {sorted(set(idx))}")

print("=== corrélation absolue (piste A) ===")
for t in (0.4, 0.5, 0.6, 0.7):
    idx, scores = ica.find_bads_eog(raw_eog, ch_name=["EOG_L","EOG_R","EOG_horiz"],
                                     threshold=t, measure="correlation", verbose=False)
    print(f"  corr {t} -> {sorted(set(idx))}")
