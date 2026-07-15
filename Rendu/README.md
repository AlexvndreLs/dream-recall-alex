# dream-recall-alex

Classification of High vs Low dream recallers (HR/LR) from overnight sleep EEG.
Replication and extension of chapter 1 of A. Dehgan's thesis
([arthurdehgan/sleep](https://github.com/arthurdehgan/sleep)).

## Installation

Requires Python >= 3.11.

```bash
pip install --pre -e .
```

The `--pre` flag is required: `specparam` (the FOOOF successor) only publishes
release candidates, so `specparam==2.0.0rc7` is not installable without it.

Dependency versions are pinned exactly to match the `mne_env` environment used
on the cluster. Results depend on the precise behaviour of specparam, pyriemann
and MNE.

## Pipeline

Four stages, run in order. Each one reads what the previous one wrote.

### 1. Raw MATLAB to BIDS

```bash
dream-bids 5 --data-path /path/to/sleep_raw_data \
             --bids-path /path/to/dream_bids
```

Reads `.mat` recordings and their hypnograms, writes a BIDS dataset
(25 channels, 1000 Hz, nose reference).

### 2. Preprocessing

```bash
dream-preprocess 5 --bids-path /path/to/dream_bids \
                   --deriv-root /path/to/dream_bids/derivatives \
                   --branches noica ica iclabel
```

Writes one BIDS derivative per branch under `derivatives/preprocessed-<branch>/`.

### 3. Feature extraction

```bash
dream-features --deriv-path /path/to/derivatives/preprocessed-noica \
               --save-path  /path/to/dream_features \
               --n-jobs     $SLURM_CPUS_PER_TASK
```

Writes cached `.npz` arrays per subject, feature and sleep stage.

### 4. Classification

```bash
dream-classify --save-path /path/to/dream_features \
               --n-jobs    $SLURM_CPUS_PER_TASK \
               --n-perm    1000 \
               --key       cov \
               --state     S2 \
               --checkpoint-every 50
```

Omitting `--key` or `--state` runs every feature or every stage.

Each command accepts `--help`. Modules can also be run directly, e.g.
`python -m dream_recall_alex.classify`.

## Repository layout

```
src/dream_recall_alex/
    config.py                    channels, subject labels, sleep stages, bands
    utils.py                     shared low-level helpers
    mat_eeg_to_bids.py           stage 1
    preprocess_subject.py        stage 2
    feat_extract_umap_fooof.py   stage 3
    classify.py                  stage 4
archive/                         Arthur's original pipeline, kept for reference
```
