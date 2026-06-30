"""Helpers partagés entre feat_extract et visualize_umap.

Ne pas mettre ici de logique métier (features, classification, plot) :
uniquement des primitives bas niveau réutilisables.
"""

from pathlib import Path

import numpy as np


def upper_tri(arr: np.ndarray) -> np.ndarray:
    """(n, p, p) -> (n, p*(p+1)/2) triangle supérieur inclusif."""
    idx = np.triu_indices(arr.shape[-1])
    return arr[..., idx[0], idx[1]].reshape(len(arr), -1)


def load_atomic(save_path: Path, key: str, sub_id: str, stage: str) -> np.ndarray | None:
    """Charge un tableau atomique caché (.npz) ou retourne None si absent."""
    f = save_path / key / f"{key}_s{str(sub_id).zfill(2)}_{stage}.npz"
    if not f.exists():
        return None
    # Fermeture explicite du fichier pour éviter le pickle error de joblib
    # (np.load retourne un NpzFile avec un BufferedReader ouvert non-picklable)
    with np.load(f) as d:
        return d["data"].copy()