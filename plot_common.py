"""Primitives partagées par les scripts de plot de la branche overlap (schéma subject).

Sépare le chargement des résultats de leur tracé : chaque plot_*.py importe d'ici
plutôt que de redupliquer la lecture des .npz et la définition des familles.

Schéma de permutation : SUBJECT (RFX) uniquement. Les figures de réplication
d'Arthur (schéma epoch) vivent dans plot_*_arthur.py et n'utilisent pas ce module.

Correction des comparaisons multiples : les p-values pooled sont lues depuis les
.npz produits par compute_maxstat_correction.py (--mode pooled), jamais
recalculées ici. La définition du pooling vit à un seul endroit : si elle change
dans compute_maxstat_correction.py, les figures suivent sans modification.
"""

from pathlib import Path

import numpy as np

from config_v3 import FREQ_DICT

# Ordre des stades pour l'affichage : les NREM groupés (S2, SWS, NREM) puis REM.
# Diffère de STATE_LIST de config_v3 (S2, SWS, REM, NREM), dont l'ordre suit
# CLASSIFICATION_GROUPS et n'a pas de logique d'affichage. Cet ordre-ci est celui
# des figures de la thèse d'Arthur.
STATES_ORDERED = ["S2", "SWS", "NREM", "REM"]

BANDS = list(FREQ_DICT)

# Familles telles que passées à compute_maxstat_correction.py --keys.
# Les noms de famille doivent matcher --family-name, sinon les .npz corrigés
# sont introuvables.
MATRIX_KEYS = ["cov"] + [f"cosp_{b}" for b in BANDS]
PSD_KEYS = [f"psd_{b}" for b in BANDS]
PSD_OSC_KEYS = [f"psd_osc_{b}" for b in BANDS]

FAMILY_KEYS = {
    "matrix": MATRIX_KEYS,
    "psd_classic": PSD_KEYS,
    "psd_osc": PSD_OSC_KEYS,
}

# Gris pour la covariance (pas une bande), puis une couleur par bande.
# Palette seaborn "deep" en dur : évite une dépendance seaborn pour 5 couleurs.
COLOR_COV = "#C2C2C2"
BAND_COLORS = {
    "delta": "#4C72B0",
    "theta": "#DD8452",
    "alpha": "#55A868",
    "sigma": "#C44E52",
    "beta": "#8172B3",
}

RESOLUTION = 300


def result_path(save_path: Path, key: str, state: str) -> Path:
    """Chemin du .npz de résultats bruts (schéma subject, pas de suffixe)."""
    return save_path / "results" / f"{key}_{state}.npz"


def load_result(save_path: Path, key: str, state: str):
    """Charge un .npz de résultats, ou None s'il est absent.

    Les combos manquants sont fréquents (runs partiels) : on retourne None pour
    laisser l'appelant décider (case vide plutôt que crash de la figure entière).
    """
    p = result_path(save_path, key, state)
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def is_matrix_key(key: str) -> bool:
    """True pour les features matricielles (1 test) vs vectorielles (19 tests).

    Détermine la forme de acc_mean : scalaire pour cov/cosp_*, vecteur (19,)
    pour psd_*, psd_osc_*, aperiodic, complexité.
    """
    return key == "cov" or key.startswith("cosp_")


def load_maxstat(corrected_path: Path, family: str, state: str) -> dict | None:
    """Charge les p-values pooled produites par compute_maxstat_correction.py.

    Retourne {test_label: pval} où test_label vaut "cosp_sigma" (matriciel) ou
    "psd_sigma/Fp2" (vectoriel), conformément au format écrit par
    compute_maxstat_correction.py. Retourne None si le fichier n'existe pas :
    à l'appelant de tracer sans marquage plutôt que d'échouer.
    """
    p = corrected_path / f"{family}_{state}_maxstat.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return dict(zip([str(x) for x in d["test_labels"]], d["pvals_corrected"]))


def load_null_max(corrected_path: Path, family: str, state: str) -> np.ndarray | None:
    """Distribution nulle du maximum sur toute la famille (n_perm,).

    C'est la loi nulle commune à tous les tests de la famille : son quantile
    (1-alpha) donne le seuil d'accuracy corrigé, directement traçable.
    """
    p = corrected_path / f"{family}_{state}_maxstat.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)["null_max"]


def maxstat_threshold(null_max: np.ndarray, alpha: float) -> float:
    """Seuil d'accuracy au quantile (1-alpha) de la loi nulle du max.

    Une accuracy au-dessus de ce seuil est significative au niveau alpha, FWER
    corrigé sur toute la famille. Formule alignée sur celle des p-values de
    compute_maxstat_correction.py : (count+1)/(n_perm+1), donc le seuil est le
    quantile empirique correspondant.
    """
    return float(np.quantile(null_max, 1 - alpha))


def band_label(key: str) -> str:
    """Étiquette lisible pour une feature : 'cov' -> 'Covariance', etc."""
    if key == "cov":
        return "Covariance"
    for prefix, suffix in (("cosp_", " cospec"), ("psd_osc_", " osc"), ("psd_", "")):
        if key.startswith(prefix):
            return key[len(prefix):] + suffix
    return key


def key_color(key: str) -> str:
    """Couleur d'une feature : gris pour cov, couleur de bande sinon."""
    if key == "cov":
        return COLOR_COV
    for band in BANDS:
        if key.endswith(band):
            return BAND_COLORS[band]
    return "#888888"