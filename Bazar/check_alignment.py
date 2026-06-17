"""
Module de vérification de l'alignement temporel des données EEG brutes.

Ce script analyse directement les fichiers .mat et .txt pour simuler
et valider la synchronisation de l'hypnogramme avec le signal EEG
avant toute conversion BIDS.
"""

import logging
from pathlib import Path
from typing import Tuple

import h5py

# Configuration du logger pour des sorties propres en console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class RawDataAlignmentVerifier:
    """
    Vérifie l'alignement temporel entre un signal EEG brut (MATLAB HDF5)
    et son hypnogramme associé (fichier texte).

    Attributes:
        eeg_path (Path): Chemin vers le fichier .mat de l'EEG.
        hypno_path (Path): Chemin vers le fichier .txt de l'hypnogramme.
        sfreq (float): Fréquence d'échantillonnage de l'EEG en Hz.
    """

    def __init__(self, eeg_path: Path, hypno_path: Path, sfreq: float = 1000.0) -> None:
        """
        Initialise le vérificateur avec les chemins des données brutes.

        Args:
            eeg_path (Path): Le chemin vers le fichier .mat.
            hypno_path (Path): Le chemin vers le fichier .txt.
            sfreq (float, optional): Fréquence d'échantillonnage. Défaut à 1000.0.
        """
        self.eeg_path = eeg_path
        self.hypno_path = hypno_path
        self.sfreq = sfreq

    def _get_eeg_duration(self) -> float:
        """
        Calcule la durée de l'EEG en lisant les métadonnées de la matrice HDF5.

        Returns:
            float: La durée de l'enregistrement en secondes.

        Raises:
            FileNotFoundError: Si le fichier .mat n'existe pas.
            KeyError: Si la variable 'm_data' n'est pas trouvée.
        """
        if not self.eeg_path.exists():
            raise FileNotFoundError(f"Fichier EEG introuvable : {self.eeg_path}")

        try:
            with h5py.File(self.eeg_path, 'r') as f:
                if 'm_data' not in f:
                    raise KeyError("La variable 'm_data' est absente du fichier .mat.")
                n_samples: int = f['m_data'].shape[0]
                return n_samples / self.sfreq
        except Exception as e:
            logging.error(f"Erreur lors de la lecture de l'EEG : {e}")
            raise

    def _get_hypno_duration(self) -> int:
        """
        Calcule la durée de l'hypnogramme en comptant le nombre de lignes.

        Returns:
            int: La durée de l'hypnogramme en secondes (1 ligne = 1 seconde).

        Raises:
            FileNotFoundError: Si le fichier texte n'existe pas.
        """
        if not self.hypno_path.exists():
            raise FileNotFoundError(f"Fichier hypnogramme introuvable : {self.hypno_path}")

        try:
            with open(self.hypno_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception as e:
            logging.error(f"Erreur lors de la lecture de l'hypnogramme : {e}")
            raise

    def analyze_alignment(self) -> Tuple[float, int, float]:
        """
        Analyse et compare les durées pour simuler l'alignement.

        Returns:
            Tuple[float, int, float]: Contient respectivement la durée de l'EEG (s),
                                      la durée de l'hypnogramme (s), et l'excédent (s).
        """
        eeg_duration: float = self._get_eeg_duration()
        hypno_duration: int = self._get_hypno_duration()
        excess_seconds: float = hypno_duration - eeg_duration

        logging.info("--- RAPPORT D'ALIGNEMENT DES DONNÉES BRUTES ---")
        logging.info(f"Sujet analysé   : {self.eeg_path.name}")
        logging.info(f"Durée EEG       : {eeg_duration:.2f} secondes")
        logging.info(f"Durée Hypno     : {hypno_duration} secondes")
        
        if excess_seconds > 0:
            logging.info(f"Conclusion      : L'hypnogramme dépasse de {excess_seconds:.2f} secondes à la fin.")
            logging.info(f"Action BIDS     : Les lignes 0 à {int(eeg_duration)} du .txt seront conservées.")
            logging.info(f"Action BIDS     : Les lignes {int(eeg_duration)} à {hypno_duration} seront ignorées.")
        elif excess_seconds < 0:
            logging.warning("Anomalie : L'EEG est plus long que l'hypnogramme !")
        else:
            logging.info("Conclusion : Les fichiers sont parfaitement synchronisés.")

        return eeg_duration, hypno_duration, excess_seconds


if __name__ == "__main__":
    # Définition des chemins vers vos données brutes spécifiques (sujet 19)
    raw_dir = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data")
    eeg_file = raw_dir / "data" / "s19_sleep.mat"
    hypno_file = raw_dir / "hypnograms" / "hyp_per_s19.txt"

    # Instanciation et exécution
    try:
        verifier = RawDataAlignmentVerifier(eeg_path=eeg_file, hypno_path=hypno_file)
        verifier.analyze_alignment()
    except Exception as error:
        logging.error(f"Le processus a échoué : {error}")