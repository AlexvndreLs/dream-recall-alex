"""Détection de saturation dans le BIDS brut (avant tout prétraitement).

Ce que ça mesure
----------------
Le BrainAmp a une plage de mesure bornée. Quand le signal la dépasse, l'ampli
ne mesure plus : il renvoie sa valeur limite (le "rail"). Une première passe a
montré que les enregistrements atteignent tous exactement 3277 µV, ce qui
correspond au rail ±3.28 mV du BrainAmp -> la saturation existe bien.

Ce script compte, par sujet et par canal, les échantillons dont la valeur
absolue atteint le rail. Il ne cherche pas de plateau strictement plat : au
rail, le bruit de quantification fait osciller la valeur d'un LSB, donc les
échantillons saturés ne sont pas rigoureusement identiques (c'est ce qui avait
fait échouer une première version de ce script).

Il rapporte aussi la durée du plus long épisode continu et le nombre d'epochs
de 30 s touchées, puisque c'est la granularité à laquelle les features sont
calculées. Le pipeline n'ayant ni AutoReject ni Potato, une epoch saturée part
telle quelle dans les features.

Comment lire le résultat
------------------------
- Aucun échantillon au rail -> pas de saturation, l'affirmation est infondée.
- Épisodes courts (< 1 s) et rares -> "ponctuel" devient mesuré plutôt
  qu'affirmé, effet probablement négligeable.
- Épisodes longs -> des epochs de 30 s sont contaminées, et elles polluent les
  PSD et les covariances.
- Comparer la liste obtenue à celle qui circulait sans source dans les
  docstrings (s5, s6, s17, s19, s20, s26, s27, s28, s37).

Usage
-----
    python detect_saturation.py --bids-path /path/to/dream_bids
    python detect_saturation.py --bids-path ... --subjects 5 6 17
"""
import argparse
from pathlib import Path

import mne_bids
import numpy as np

from dream_recall_alex.config import CH_NAMES, EPOCH_DURATION, N_EEG

# Rail du BrainAmp, en µV. La valeur observée sur les données est 3277 µV ; on
# prend une marge d'un LSB (0.1 µV) pour attraper l'oscillation de
# quantification autour du rail.
RAIL_UV = 3277.0
RAIL_TOL_UV = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bids-path", type=Path, required=True,
                   help="Racine du BIDS produit par mat_eeg_to_bids")
    p.add_argument("--subjects", type=int, nargs="*", default=None,
                   help="Sujets à analyser (défaut : 1 à 38)")
    p.add_argument("--rail", type=float, default=RAIL_UV,
                   help=f"Rail de l'ampli en µV (défaut {RAIL_UV:.0f})")
    return p.parse_args()


def episodes(mask: np.ndarray) -> np.ndarray:
    """Longueurs des épisodes contigus de True dans `mask`."""
    if not mask.any():
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return edges[1::2] - edges[0::2]


def analyse_subject(bids_path: Path, sub_id: int, rail_v: float):
    """Compte les échantillons au rail pour un sujet. None si absent du BIDS."""
    sub_str = str(sub_id).zfill(2)
    bp = mne_bids.BIDSPath(subject=sub_str, task="sleep",
                           root=bids_path, datatype="eeg")
    try:
        raw = mne_bids.read_raw_bids(bp, verbose="ERROR")
    except Exception:
        return None

    raw.load_data(verbose="ERROR")
    raw.pick(CH_NAMES[:N_EEG])
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    n_times = data.shape[1]

    saturated = np.abs(data) >= (rail_v - RAIL_TOL_UV * 1e-6)

    per_channel = {}
    for ch_idx in range(data.shape[0]):
        m = saturated[ch_idx]
        if not m.any():
            continue
        ep = episodes(m)
        per_channel[CH_NAMES[ch_idx]] = dict(
            n_samples=int(m.sum()),
            pct=100 * m.sum() / n_times,
            n_episodes=int(len(ep)),
            longest_s=float(ep.max() / sfreq),
        )

    # Epochs de 30 s touchées par au moins un échantillon saturé, sur au moins
    # un canal : c'est la granularité à laquelle les features sont calculées.
    n_per_epoch = int(EPOCH_DURATION * sfreq)
    n_epochs = n_times // n_per_epoch
    if n_epochs:
        any_ch = saturated.any(axis=0)[: n_epochs * n_per_epoch]
        epochs_touched = int(any_ch.reshape(n_epochs, n_per_epoch).any(axis=1).sum())
    else:
        epochs_touched = 0

    return dict(
        sub_id=sub_id,
        duration_h=n_times / sfreq / 3600,
        amplitude_max_uV=float(np.abs(data).max()) * 1e6,
        n_channels_touched=len(per_channel),
        per_channel=per_channel,
        n_epochs=n_epochs,
        epochs_touched=epochs_touched,
        total_saturated_s=float(saturated.any(axis=0).sum() / sfreq),
    )


def main():
    args = parse_args()
    rail_v = args.rail * 1e-6
    subjects = args.subjects or list(range(1, 39))

    print(f"BIDS   : {args.bids_path}")
    print(f"Rail   : {args.rail:.0f} µV (± {RAIL_TOL_UV:.1f} µV)")
    print(f"Sujets : {len(subjects)} à analyser\n")

    results = []
    for sub_id in subjects:
        r = analyse_subject(args.bids_path, sub_id, rail_v)
        if r is None:
            print(f"  s{sub_id:02d} : absent du BIDS")
            continue
        results.append(r)
        if r["n_channels_touched"] == 0:
            print(f"  s{sub_id:02d} : rien              "
                  f"({r['duration_h']:.1f}h, max {r['amplitude_max_uV']:.0f} µV)")
        else:
            worst = max(r["per_channel"].values(), key=lambda d: d["n_samples"])
            pct_ep = 100 * r["epochs_touched"] / r["n_epochs"] if r["n_epochs"] else 0
            print(
                f"  s{sub_id:02d} : {r['n_channels_touched']:2d} canaux, "
                f"{r['total_saturated_s']:8.2f}s saturées, "
                f"épisode max {worst['longest_s']:6.2f}s, "
                f"{r['epochs_touched']:4d}/{r['n_epochs']} epochs ({pct_ep:5.2f}%)"
            )

    # ─── synthèse ────────────────────────────────────────────────────────────
    touched = [r for r in results if r["n_channels_touched"] > 0]
    print(f"\n{'=' * 78}")
    print(f"{len(touched)}/{len(results)} sujets avec au moins un échantillon au rail")

    if not touched:
        print("\nAucune saturation détectée à ce seuil.")
        return

    print("\nSujets concernés :", ", ".join(f"s{r['sub_id']}" for r in touched))

    claimed = {5, 6, 17, 19, 20, 26, 27, 28, 37}
    found = {r["sub_id"] for r in touched}
    print(f"\nComparaison à la liste non sourcée {sorted(claimed)} :")
    print(f"  dans les deux      : {sorted(found & claimed)}")
    print(f"  trouvés seulement  : {sorted(found - claimed)}")
    print(f"  annoncés seulement : {sorted(claimed - found)}")

    tot_ep = sum(r["epochs_touched"] for r in touched)
    tot_all = sum(r["n_epochs"] for r in results)
    if tot_all:
        print(f"\nEpochs de {EPOCH_DURATION:.0f}s touchées : {tot_ep} sur {tot_all} "
              f"({100 * tot_ep / tot_all:.2f}%)")
        print("Ces epochs partent telles quelles dans les features : le pipeline")
        print("n'a ni AutoReject ni Potato.")

    longest = max(d["longest_s"] for r in touched for d in r["per_channel"].values())
    print(f"\nPlus long épisode continu : {longest:.2f}s")


if __name__ == "__main__":
    main()
