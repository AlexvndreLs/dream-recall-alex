import itertools
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Set


class SleepMacroAnalyzer:
    """Classe pour analyser l'architecture globale d'un hypnogramme.

    Extrait les statistiques de durée et compte les transitions (blocs continus)
    pour chaque code brut du fichier.
    """

    def __init__(self, file_path: str | Path) -> None:
        """Initialise l'analyseur avec le chemin du fichier cible.

        Args:
            file_path (str | Path): Chemin absolu vers le fichier hypnogramme texte.
        """
        self.file_path = Path(file_path)
        self.hypnogram: np.ndarray = self._load_data()
        
        # Dictionnaire des codes bruts uniquement
        self.labels = {k: str(k) for k in range(-2, 6)}

    def _load_data(self) -> np.ndarray:
        """Charge les données de l'hypnogramme en mémoire.
        
        Returns:
            np.ndarray: Vecteur des codes de l'hypnogramme.
            
        Raises:
            FileNotFoundError: Si le fichier cible n'existe pas.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Le fichier {self.file_path} est introuvable.")
            
        with open(self.file_path, "r", encoding="utf-8") as file_pointer:
            data = [int(line.strip()) for line in file_pointer if line.strip()]
        return np.array(data)

    def extract_continuous_blocks(self) -> Dict[int, List[int]]:
        """Identifie et mesure tous les blocs continus pour chaque code.

        Returns:
            Dict[int, List[int]]: Un dictionnaire où la clé est le code,
            et la valeur est une liste contenant la durée (en secondes).
        """
        blocks_dict: Dict[int, List[int]] = {k: [] for k in self.labels.keys()}
        
        for code, group in itertools.groupby(self.hypnogram):
            duration = len(list(group))
            if code in blocks_dict:
                blocks_dict[code].append(duration)
                
        return blocks_dict

    def generate_report_string(self) -> str:
        """Génère le rapport statistique formaté sous forme de chaîne de caractères.

        Returns:
            str: Le bloc de texte complet contenant les statistiques du fichier.
        """
        blocks_dict = self.extract_continuous_blocks()
        total_seconds = len(self.hypnogram)
        
        lines: List[str] = []
        lines.append(f"\n{'='*80}")
        lines.append(f"RAPPORT D'ARCHITECTURE : {self.file_path.name}")
        lines.append(f"Durée totale de l'enregistrement : {total_seconds / 3600:.2f} heures")
        lines.append(f"{'='*80}\n")
        
        lines.append(f"{'Code':<6} | {'% Nuit':<8} | {'Total (min)':<12} | {'Nb Blocs':<10} | {'Moy (s)':<8} | {'Min (s)':<8} | {'Max (s)':<8}")
        lines.append("-" * 80)
        
        for code in sorted(self.labels.keys()):
            durations = blocks_dict[code]
            code_str = self.labels[code]
            
            if not durations:
                lines.append(f"{code_str:<6} | {'0.0%':<8} | {'0.0':<12} | {'0':<10} | {'-':<8} | {'-':<8} | {'-':<8}")
                continue
                
            total_time_sec = sum(durations)
            percentage = (total_time_sec / total_seconds) * 100
            total_time_min = total_time_sec / 60
            
            n_blocks = len(durations)
            mean_d = np.mean(durations)
            min_d = np.min(durations)
            max_d = np.max(durations)
            
            lines.append(f"{code_str:<6} | {percentage:>5.1f}%   | {total_time_min:>9.1f}   | {n_blocks:>8}   | {mean_d:>7.1f}  | {min_d:>7}  | {max_d:>7}")

        lines.append("\n" + "="*80)
        return "\n".join(lines)


def extract_all_subject_ids(directory: Path) -> List[int]:
    """Scanne le dossier et extrait une liste unique et triée de tous les numéros de sujets.
    
    Args:
        directory (Path): Le dossier contenant les hypnogrammes.
        
    Returns:
        List[int]: Une liste d'entiers triés (ex: [1, 2, 3, 4, 10, 11...]).
    """
    subject_ids: Set[int] = set()
    
    for filepath in directory.glob("hyp_*.txt"):
        # On cherche un 's' suivi de chiffres (ex: s1, s10)
        match = re.search(r's(\d+)', filepath.name)
        if match:
            subject_ids.add(int(match.group(1)))
            
    return sorted(list(subject_ids))


# --- Zone d'exécution ---
if __name__ == "__main__":
    HYP_DIR = Path("/project/rrg-kjerbi/shared/dream_recall/sleep_data/sleep_raw_data/hypnograms")
    OUTPUT_FILE = Path("./rapport_architecture_complet.txt")
    
    # 1. Extraction et tri de tous les numéros de sujets disponibles
    sorted_subject_ids = extract_all_subject_ids(HYP_DIR)
    
    # 2. Création de la liste ordonnée par paires (JBE puis PER)
    ordered_files: List[Path] = []
    for sub_id in sorted_subject_ids:
        file_jbe = HYP_DIR / f"hyp_jbe_s{sub_id}.txt"
        file_per = HYP_DIR / f"hyp_per_s{sub_id}.txt"
        
        # On ajoute à la liste de traitement uniquement si le fichier existe
        if file_jbe.exists():
            ordered_files.append(file_jbe)
        if file_per.exists():
            ordered_files.append(file_per)
            
    # 3. Traitement silencieux et écriture
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        out_file.write("=== RAPPORT BATCH : COMPARAISON JBE VS PER PAR SUJET ===\n")
        
        for file_path in ordered_files:
            try:
                analyzer = SleepMacroAnalyzer(file_path=file_path)
                report_str = analyzer.generate_report_string()
                out_file.write(report_str + "\n")
            except Exception as e:
                out_file.write(f"\n[!] ERREUR lors de l'analyse de {file_path.name} : {e}\n")
                
    # Output console minimal
    print(f"Terminé ! Les {len(ordered_files)} fichiers ont été sauvegardés en paires dans : {OUTPUT_FILE.absolute()}")