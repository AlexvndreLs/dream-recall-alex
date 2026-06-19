"""Script d'inspection des composantes ICA (sauvegarde PNG).

À lancer après le preprocessing pour valider que les composantes rejetées
par find_bads_eog / find_bads_muscle sont bien des artefacts, et calibrer le
seuil threshold de find_bads_muscle si nécessaire.

Les figures sont SAUVÉES EN PNG (backend Agg, non-interactif) -> exécutable
sur un nœud de cluster sans écran (Fir login). Récupérer les PNG produits
dans --out-dir et les inspecter localement.

Prérequis :
    - preprocess_subject_v2.py doit avoir tourné pour le sujet cible
    - L'objet ICA est dans derivatives/ica/sub-XX_task-sleep_ica.fif
    - Le BIDS brut est dans dream_bids/ (25 canaux, 1000Hz)

Usage :
    python inspect_ica.py 5 \\
        --bids-path  /home/alouis/scratch/dream_bids \\
        --deriv-root /home/alouis/scratch/dream_bids/derivatives \\
        --out-dir    ./ica_figures

Ce que fait ce script :
    1. Charge le raw BIDS brut (25 canaux, 1000Hz)
    2. Recharge l'objet ICA sauvegardé
    3. Sauve les composantes rejetées (topographies, propriétés détaillées)
    4. Sauve toutes les composantes pour comparaison
    5. Sauve le signal EEG avant/après ICA

Les figures sont écrites en PNG dans --out-dir (pas d'affichage interactif).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend non-interactif : sauvegarde PNG sans écran (cluster)
import matplotlib.pyplot as plt

import mne
import mne_bids

from config_v3 import CH_NAMES, HP_FREQ_ICA, N_EEG


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=int)
    p.add_argument('--bids-path', type=Path, required=True,
                   help="Racine du BIDS brut (dream_bids/)")
    p.add_argument('--deriv-root', type=Path, required=True,
                   help="Racine derivatives (contient ica/)")
    p.add_argument('--hp-freq', type=float, default=HP_FREQ_ICA,
                   help=f"HP pour la visualisation ICA (défaut: {HP_FREQ_ICA}Hz)")
    p.add_argument('--out-dir', type=Path, default=Path("./ica_figures"),
                   help="Dossier de sortie des PNG (défaut: ./ica_figures)")
    p.add_argument('--full', action="store_true", default=False,
                   help="Ajoute les figures temporelles lourdes (sources + signal "
                        "avant/après sur 9h) : lent, illisible, hors décision. Off par défaut.")
    return p.parse_args()


def save_figs(figs, out_dir: Path, name: str) -> None:
    """Sauve une figure ou une liste de figures en PNG dans out_dir.

    Les plot_* de MNE renvoient soit une figure, soit une liste (ex:
    plot_properties -> une figure par composante). On gère les deux et on
    suffixe par un index si plusieurs.
    """
    if not isinstance(figs, (list, tuple)):
        figs = [figs]
    for i, fig in enumerate(figs):
        suffix = f"_{i}" if len(figs) > 1 else ""
        path = out_dir / f"{name}{suffix}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args    = parse_args()
    sub_str = str(args.subject).zfill(2)

    # 1. Charge le raw BIDS brut
    bids_input = mne_bids.BIDSPath(
        subject=sub_str, task='sleep',
        root=args.bids_path, datatype='eeg',
    )
    raw = mne_bids.read_raw_bids(bids_input, verbose=False)
    raw.load_data()

    # HP 1Hz pour visualisation cohérente avec le fit ICA
    raw_vis = raw.copy().filter(l_freq=args.hp_freq, h_freq=None, verbose=False)

    # 2. Recharge l'objet ICA
    ica_path = args.deriv_root / "ica" / f"sub-{sub_str}_task-sleep_ica.fif"
    if not ica_path.exists():
        raise FileNotFoundError(
            f"ICA non trouvé : {ica_path}\n"
            "Lancer preprocess_subject_v2.py d'abord."
        )
    ica = mne.preprocessing.read_ica(ica_path)

    print(f"sub-{sub_str} — {ica.n_components_} composantes ICA")
    print(f"Composantes exclues automatiquement : {ica.exclude}")

    out_dir = args.out_dir / f"sub-{sub_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Topographies + propriétés des composantes exclues
    if ica.exclude:
        print("\n--- Composantes rejetées (topographies) ---")
        figs = ica.plot_components(picks=ica.exclude, title=f"sub-{sub_str} — rejetées", show=False)
        save_figs(figs, out_dir, "rejected_topo")

        print("\n--- Propriétés détaillées des composantes rejetées ---")
        figs = ica.plot_properties(raw_vis, picks=ica.exclude, show=False)
        save_figs(figs, out_dir, "rejected_props")
    else:
        print("Aucune composante rejetée automatiquement.")

    # 4. Toutes les composantes pour comparaison
    print("\n--- Toutes les composantes (sources temporelles) ---")
    figs = ica.plot_sources(raw_vis, title=f"sub-{sub_str} — toutes composantes", show=False)
    save_figs(figs, out_dir, "all_sources")

    print("\n--- Toutes les topographies ---")
    figs = ica.plot_components(title=f"sub-{sub_str} — toutes topographies", show=False)
    save_figs(figs, out_dir, "all_topo")

    # Figures temporelles lourdes (9h × 1000Hz) : lentes et illisibles compressées
    # sur une largeur d'écran, hors décision de validation des artefacts.
    # Désactivées par défaut -> évite de saturer le temps d'un job interactif.
    if args.full:
        print("\n--- Toutes les composantes (sources temporelles) ---")
        figs = ica.plot_sources(raw_vis, title=f"sub-{sub_str} — toutes composantes", show=False)
        save_figs(figs, out_dir, "all_sources")

        # Signal avant/après ICA sur les 19 EEG
        raw_eeg = raw_vis.copy().pick(CH_NAMES[:N_EEG])
        raw_clean = ica.apply(raw_eeg.copy(), verbose=False)

        print("\n--- Signal EEG avant ICA ---")
        fig = raw_eeg.plot(title=f"sub-{sub_str} — avant ICA", scalings='auto', show=False)
        save_figs(fig, out_dir, "signal_before")

        print("\n--- Signal EEG après ICA ---")
        fig = raw_clean.plot(title=f"sub-{sub_str} — après ICA", scalings='auto', show=False)
        save_figs(fig, out_dir, "signal_after")

    print(f"\nInspection terminée. Figures dans : {out_dir}")
    print("Pour modifier les composantes rejetées, éditer ica.exclude manuellement")
    print("et relancer preprocess_subject_v2.py avec les nouveaux paramètres.")
