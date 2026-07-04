"""Compte le nombre d'epochs de 30s par sujet et par stade, EXACTEMENT selon
la meme logique que load_epochs_by_atomic_stage() dans
feat_extract_umap_fooof_v4.py (blocs de 30 annotations consecutives espacees
de SF samples, meme stade) -- mais sans charger les donnees EEG, juste
events.tsv, donc rapide (pas besoin de sbatch, tourne en quelques secondes
en interactif).

But : comparer nos comptages avec Riemannian_Dream_Recall_Subject_numbers.xlsx
(envoye par Arthur) pour verifier independamment que le scoring/segmentation
est coherent avec le sien, sujet par sujet.

Usage:
    python count_epochs_per_subject.py \\
        --deriv-path /home/alouis/scratch/dream_bids/derivatives_1000hz/preprocessed-noica
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config_v3 import SUBJECT_IDS_ANALYSIS, STAGE_LABEL_TO_ATOMIC, ATOMIC_STAGES, PER_BLACKLIST_STR, JBE_SUBJECTS_STR, SFREQ_PREPROC

SF = int(SFREQ_PREPROC)


def _events(deriv_path: Path, sub_id: str) -> Path:
    return (deriv_path / f"sub-{sub_id}" / "eeg"
            / f"sub-{sub_id}_task-sleep_proc-clean_events.tsv")


def _choose_scorer(sub_id: str) -> str:
    if sub_id not in PER_BLACKLIST_STR:
        return "per"
    if sub_id in JBE_SUBJECTS_STR:
        return "jbe"
    raise ValueError(f"sub-{sub_id}: no valid scorer")


def count_epochs(deriv_path: Path, sub_id: str) -> dict[str, int]:
    ev_path = _events(deriv_path, sub_id)
    if not ev_path.exists():
        print(f"  MANQUANT : {ev_path}")
        return {}

    scorer = _choose_scorer(sub_id)
    prefix = f"{scorer}/"

    df = pd.read_csv(ev_path, sep="\t")
    df = df[df["trial_type"].str.startswith(prefix)].copy()
    df["stage"] = df["trial_type"].str[len(prefix):]
    df = (df[df["stage"].isin(STAGE_LABEL_TO_ATOMIC)]
          .sort_values("sample")
          .reset_index(drop=True))

    counts = {s: 0 for s in ATOMIC_STAGES}
    i = 0
    while i + 29 < len(df):
        block = df.iloc[i:i + 30]
        samples = block["sample"].values
        stages = block["stage"].values
        if not (np.all(samples == samples[0] + np.arange(30) * SF) and
                np.all(stages == stages[0])):
            i += 1
            continue
        counts[STAGE_LABEL_TO_ATOMIC[stages[0]]] += 1
        i += 30
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-path", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for sub_id in SUBJECT_IDS_ANALYSIS:
        c = count_epochs(args.deriv_path, sub_id)
        if not c:
            continue
        sws = c.get("S3", 0) + c.get("S4", 0)
        nrem = c.get("S1", 0) + c.get("S2", 0) + c.get("S3", 0) + c.get("S4", 0)
        rows.append(dict(
            subject=int(sub_id), S1=c.get("S1", 0), S2=c.get("S2", 0),
            SWS=sws, REM=c.get("REM", 0), NREM=nrem,
        ))

    out = pd.DataFrame(rows).sort_values("subject")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(out.to_string(index=False))
    print()
    print("Colle ce tableau a cote de Riemannian_Dream_Recall_Subject_numbers.xlsx")
    print("pour comparer sujet par sujet, stade par stade.")
