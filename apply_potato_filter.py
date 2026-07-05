"""Filtre les matrices atomiques (cov, cosp_*) avec un Riemannian Potato
(pyriemann.clustering.Potato), pour rejeter les epochs excentriques
(artefacts résiduels) avant classification.

NE TOUCHE JAMAIS aux données originales : lit dream_features_noica/{key}/,
écrit dream_features_noica_potato/{key}/ (nouveau dossier, même format
.npz atomique) — classify.py n'a besoin d'aucune modification, il suffit
de le relancer avec --save-path pointant vers le nouveau dossier.

Le Potato est ajusté PAR SUJET PAR STADE (pas globalement), car la
distribution "normale" des matrices de covariance varie d'un sujet à
l'autre (impédance, montage, etc.) — un seuil global mélangerait cette
variabilité inter-sujet avec les vrais artefacts intra-sujet qu'on veut
détecter.

Usage :
    python apply_potato_filter.py \\
        --save-path-in  /home/alouis/scratch/dream_features_noica \\
        --save-path-out /home/alouis/scratch/dream_features_noica_potato \\
        --threshold 3.0
"""
import argparse
import time
from pathlib import Path

import numpy as np
from pyriemann.clustering import Potato

MATRIX_KEYS = ["cov", "cosp_delta", "cosp_theta", "cosp_alpha", "cosp_sigma", "cosp_beta"]
STATES = ["S1", "S2", "SWS", "REM"]  # stages atomiques (avant regroupement CLASSIFICATION_GROUPS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path-in", type=Path, required=True)
    p.add_argument("--save-path-out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=3.0,
                    help="Seuil z-score riemannien au-dela duquel une epoch est rejetee (defaut pyriemann: 3.0)")
    p.add_argument("--n-iter-max", type=int, default=300,
                    help="Iterations max pour le calcul de la moyenne riemannienne interne au Potato "
                         "(defaut pyriemann: 100 -> augmente a 300 suite a 1/189 non-convergences observees)")
    p.add_argument("--keys", nargs="+", default=MATRIX_KEYS,
                    help="Features a filtrer (matricielles uniquement, Potato opere sur des matrices SPD)")
    return p.parse_args()


def _load_with_retry(path: Path, n_retries: int = 3, delay: float = 2.0):
    """Charge un .npz avec retry — Lustre peut renvoyer des erreurs I/O
    transitoires (Errno 5) sous forte contention (plusieurs jobs concurrents
    lisant/écrivant en même temps sur le même filesystem partagé)."""
    for attempt in range(n_retries):
        try:
            return np.load(path)
        except OSError as e:
            if attempt == n_retries - 1:
                raise
            print(f"    retry lecture {path.name} ({attempt+1}/{n_retries}) apres: {e}")
            time.sleep(delay)


def _save_with_retry(path: Path, data, n_retries: int = 3, delay: float = 2.0):
    """Sauvegarde un .npz avec retry, meme raison que _load_with_retry."""
    for attempt in range(n_retries):
        try:
            np.savez_compressed(path, data=data)
            return
        except OSError as e:
            if attempt == n_retries - 1:
                raise
            print(f"    retry ecriture {path.name} ({attempt+1}/{n_retries}) apres: {e}")
            time.sleep(delay)


def filter_subject_key_stage(in_path: Path, out_path: Path, threshold: float, n_iter_max: int = 300) -> tuple[int, int]:
    """Ajuste un Potato sur les matrices d'un fichier .npz atomique, filtre, sauvegarde.

    Retourne (n_avant, n_apres) pour le rapport.
    """
    d = _load_with_retry(in_path)
    mats = d["data"]  # shape (n_epochs, n_channels, n_channels)
    n_before = len(mats)

    if n_before < 10:
        # Trop peu d'epochs pour ajuster un Potato de façon fiable : on garde tel quel.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_with_retry(out_path, mats)
        return n_before, n_before

    potato = Potato(threshold=threshold, n_iter_max=n_iter_max)
    potato.fit(mats)
    labels = potato.predict(mats)  # 1 = inlier, 0 = outlier
    kept = mats[labels == 1]

    # Garde-fou : si le Potato rejette trop d'epochs (>50%), le seuil est
    # probablement trop agressif pour ce sujet — on garde les données brutes
    # plutôt que de risquer de casser n_trials_min pour tout le monde.
    if len(kept) < n_before * 0.5:
        kept = mats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_with_retry(out_path, kept)
    return n_before, len(kept)


if __name__ == "__main__":
    args = parse_args()
    args.save_path_out.mkdir(parents=True, exist_ok=True)

    total_before, total_after = 0, 0
    report = []

    for key in args.keys:
        in_dir = args.save_path_in / key
        if not in_dir.exists():
            print(f"SKIP {key} : dossier absent dans {args.save_path_in}")
            continue
        for f in sorted(in_dir.glob("*.npz")):
            out_file = args.save_path_out / key / f.name
            try:
                n_before, n_after = filter_subject_key_stage(f, out_file, args.threshold, args.n_iter_max)
                total_before += n_before
                total_after += n_after
                pct_kept = 100 * n_after / n_before if n_before else 100
                report.append((key, f.stem, n_before, n_after, pct_kept))
                if pct_kept < 90:
                    print(f"  {key}/{f.stem} : {n_before} -> {n_after} epochs ({pct_kept:.0f}% gardees)")
            except Exception as e:
                print(f"ERREUR {f} : {e}")

    print()
    print(f"=== Total : {total_before} -> {total_after} epochs ({100*total_after/total_before:.1f}% gardees) ===")
    print(f"Sortie : {args.save_path_out}")
    print()
    print("Prochaine etape : relancer classify.py avec --save-path pointant vers le dossier _potato")
