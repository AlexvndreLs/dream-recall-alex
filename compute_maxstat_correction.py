"""Calcule les p-values corrigées (max-stat) en regroupant les perm_accs
de plusieurs features au sein d'un même stade — méthode Arthur §1.2.8 :
"maximum statistics ... across electrodes or frequency bands".

NE TOUCHE JAMAIS aux fichiers originaux : lit {save-path}/results/*.npz,
écrit dans {output-path}/*.npz (nouveau dossier, un fichier par famille x
stade).

Prérequis (déjà en place suite au BUGFIX seed du 01/07) : les perm_accs de
toutes les features d'une même famille doivent avoir été calculées avec le
MÊME seed de permutation par indice p (seed basé sur ('perm', state, ...),
pas sur key) — sinon "le max à l'itération p" n'a aucun sens statistique
cohérent entre features.

Deux modes de correction disponibles (--mode) :

  pooled (défaut, notre méthode) : empile TOUTES les keys passées à --keys
    ensemble (ex: les 5 bandes PSD x 19 électrodes = 95 tests) et prend le
    max sur ce pool complet. Correspond au texte de la thèse §1.2.9
    ("maximum statistics ... across electrodes AND frequency bands").

  arthur (réplique exacte du code) : reproduit maxstat_pval.py du repo
    github.com/arthurdehgan/sleep — le max-stat est calculé PAR KEY
    (une bande à la fois), sur les électrodes SEULEMENT. Pas de pooling
    inter-bandes. C'est ce que le CODE d'Arthur fait réellement, même si
    le texte de la thèse suggère une correction plus large. Vérifié
    empiriquement le 04/07/2026 en lisant maxstat_pval.py sur son repo.
    Différence de formule mineure aussi notée : Arthur fait
    count/(N_PERM+1) avec N_PERM=999 (donc /1000... en fait /1001 vu le
    commentaire "1/1001" dans son code, à priori une coquille pour 1000
    perms + 1) ; nous faisons (count+1)/(n_perm+1) avec n_perm=1000.
    Écart marginal, non corrigé ici pour rester sur une formule standard.

Usage :
    # Mode pooled (notre méthode) : famille matricielle (cov + 5 cosp), comme Arthur Fig.2
    python compute_maxstat_correction.py \\
        --save-path /home/alouis/scratch/dream_features_noica \\
        --output-path /home/alouis/scratch/dream_features_noica_corrected \\
        --family-name matrix \\
        --keys cov cosp_delta cosp_theta cosp_alpha cosp_sigma cosp_beta

    # Mode pooled : famille PSD classique (5 bandes x 19 electrodes = 95 tests), comme Arthur Fig.4
    python compute_maxstat_correction.py \\
        --save-path /home/alouis/scratch/dream_features_noica \\
        --output-path /home/alouis/scratch/dream_features_noica_corrected \\
        --family-name psd_classic \\
        --keys psd_delta psd_theta psd_alpha psd_sigma psd_beta

    # Feature exploratoire isolee (corrigee seulement sur ses 19 electrodes)
    python compute_maxstat_correction.py \\
        --save-path /home/alouis/scratch/dream_features_noica \\
        --output-path /home/alouis/scratch/dream_features_noica_corrected \\
        --family-name higuchi_fd \\
        --keys higuchi_fd

    # Mode arthur : réplique exacte maxstat_pval.py, une bande a la fois
    # (traite chaque key independamment, --family-name ignore dans ce mode)
    python compute_maxstat_correction.py \\
        --save-path /home/alouis/scratch/dream_features_noica_1000hz \\
        --output-path /home/alouis/scratch/dream_features_noica_1000hz_corrected \\
        --family-name unused --mode arthur \\
        --keys psd_delta psd_theta psd_alpha psd_sigma psd_beta
"""
import argparse
from pathlib import Path

import numpy as np

STATES = ["S2", "SWS", "NREM", "REM"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True,
                    help="Dossier contenant results/*.npz (deja calcules par classify.py)")
    p.add_argument("--output-path", type=Path, required=True,
                    help="Dossier de SORTIE pour les resultats corriges (jamais le meme que save-path)")
    p.add_argument("--family-name", type=str, required=True,
                    help="Nom de la famille de tests (utilise dans le nom du fichier de sortie). "
                         "Ignore en mode 'arthur' (un fichier par key genere a la place).")
    p.add_argument("--keys", nargs="+", required=True,
                    help="Features a regrouper pour la correction (ex: les 6 features matricielles, "
                         "ou les 5 bandes PSD classiques, ou une seule feature isolee). "
                         "En mode 'arthur', chaque key est traitee independamment.")
    p.add_argument("--states", nargs="+", default=STATES)
    p.add_argument("--mode", choices=["pooled", "arthur"], default="pooled",
                    help="pooled: notre methode (max sur toutes les keys+electrodes ensemble, "
                         "un seul pool). "
                         "arthur: replique maxstat_pval.py du repo d'Arthur (max sur electrodes "
                         "seulement, 1 key/bande a la fois, pas de pooling inter-bandes).")
    return p.parse_args()


def load_combo(results_dir: Path, key: str, state: str):
    f = results_dir / f"{key}_{state}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return d


def correct_one_state(results_dir: Path, keys: list, state: str) -> dict | None:
    """Regroupe les perm_accs de `keys` pour `state`, calcule le max-stat.

    Mode 'pooled' (notre methode) : toutes les keys passees sont empilees dans
    un seul pool avant de prendre le max. Retourne un dict pret a sauvegarder,
    ou None si des donnees manquent.
    """
    loaded = {}
    for key in keys:
        d = load_combo(results_dir, key, state)
        if d is None:
            print(f"  MANQUANT : {key}_{state}.npz — famille incomplete pour {state}, skip")
            return None
        if "perm_accs" not in d:
            print(f"  MANQUANT : perm_accs absent dans {key}_{state}.npz — skip")
            return None
        loaded[key] = d

    # Verification n_perm coherent entre toutes les features de la famille
    n_perms = {key: len(d["perm_accs"]) for key, d in loaded.items()}
    if len(set(n_perms.values())) > 1:
        print(f"  ATTENTION : n_perm incoherent entre features pour {state} : {n_perms} — skip")
        return None
    n_perm = next(iter(n_perms.values()))

    # Empile tous les perm_accs de la famille en une seule matrice
    # (n_perm, n_tests_total) — n_tests_total = somme des tests par feature
    # (1 pour matrice, n_elec pour vecteur).
    perm_pool = []   # liste de colonnes (n_perm,) a concatener
    real_pool = []   # liste de valeurs reelles alignees avec perm_pool
    test_labels = [] # (key, electrode_ou_None) pour tracer chaque colonne

    for key, d in loaded.items():
        perm_accs = d["perm_accs"]
        acc_mean = d["acc_mean"]
        if perm_accs.ndim == 1:
            # feature matricielle : 1 seul test
            perm_pool.append(perm_accs.reshape(-1, 1))
            real_pool.append(float(acc_mean))
            test_labels.append((key, None))
        else:
            # feature vectorielle : n_elec tests
            perm_pool.append(perm_accs)  # deja (n_perm, n_elec)
            ch_names = d["ch_names"] if "ch_names" in d else [str(i) for i in range(perm_accs.shape[1])]
            for e, ch in enumerate(ch_names):
                real_pool.append(float(acc_mean[e]))
                test_labels.append((key, str(ch)))

    perm_matrix = np.concatenate(perm_pool, axis=1)  # (n_perm, n_tests_total)
    real_values = np.array(real_pool)                # (n_tests_total,)

    # Coeur du max-stat : pour chaque permutation, le max sur TOUS les tests
    # de la famille -> une distribution nulle commune, partagee par tous les
    # tests de cette famille.
    null_max = perm_matrix.max(axis=1)  # (n_perm,)

    pvals_corrected = (np.sum(null_max[:, None] >= real_values[None, :], axis=0) + 1) / (n_perm + 1)

    return dict(
        keys=np.array(list(loaded.keys())),
        test_labels=np.array([f"{k}/{c}" if c else k for k, c in test_labels]),
        real_values=real_values,
        pvals_corrected=pvals_corrected,
        null_max=null_max,
        n_perm=n_perm,
        n_tests=len(real_values),
    )


def correct_one_key_arthur_style(results_dir: Path, key: str, state: str) -> dict | None:
    """Replique exactement maxstat_pval.py d'Arthur (github.com/arthurdehgan/sleep) :
    max-stat sur les electrodes SEULEMENT, une feature/bande a la fois (pas de
    pooling inter-bandes/inter-keys). Ne s'applique qu'aux features vectorielles
    (perm_accs de forme (n_perm, n_elec)) — n'a pas de sens pour une feature
    matricielle qui n'a qu'un seul test (rien sur quoi prendre un max).
    """
    d = load_combo(results_dir, key, state)
    if d is None:
        print(f"  MANQUANT : {key}_{state}.npz — skip")
        return None
    if "perm_accs" not in d:
        print(f"  MANQUANT : perm_accs absent dans {key}_{state}.npz — skip")
        return None
    perm_accs = d["perm_accs"]
    if perm_accs.ndim != 2:
        print(f"  {key}_{state} : perm_accs n'est pas (n_perm, n_elec) -> mode 'arthur' "
              f"ne s'applique qu'aux features vectorielles, skip")
        return None
    acc_mean = d["acc_mean"]
    ch_names = d["ch_names"] if "ch_names" in d else [str(i) for i in range(perm_accs.shape[1])]
    n_perm = perm_accs.shape[0]

    # Coeur exact d'Arthur (maxstat_pval.py) : max sur les electrodes seulement,
    # pour CETTE bande/key uniquement (pas de pool avec d'autres bandes).
    null_max = perm_accs.max(axis=1)  # (n_perm,)
    pvals_corrected = (np.sum(null_max[:, None] >= acc_mean[None, :], axis=0) + 1) / (n_perm + 1)

    return dict(
        key=key,
        ch_names=np.array(ch_names),
        real_values=acc_mean,
        pvals_corrected=pvals_corrected,
        null_max=null_max,
        n_perm=n_perm,
        n_tests=len(acc_mean),
    )


if __name__ == "__main__":
    args = parse_args()
    results_dir = args.save_path / "results"
    args.output_path.mkdir(parents=True, exist_ok=True)

    for state in args.states:
        if args.mode == "arthur":
            for key in args.keys:
                print(f"=== {key} x {state} (mode arthur, max-stat electrodes seulement) ===")
                result = correct_one_key_arthur_style(results_dir, key, state)
                if result is None:
                    continue

                out_file = args.output_path / f"{key}_{state}_maxstat_arthur.npz"
                np.savez_compressed(out_file, **result)

                n_sig = int((result["pvals_corrected"] < 0.05).sum())
                print(f"  {result['n_tests']} electrodes, {result['n_perm']} perms")
                print(f"  {n_sig}/{result['n_tests']} significatifs a p<0.05 (corrige, mode arthur)")
                best_idx = result["pvals_corrected"].argmin()
                print(f"  meilleur : {result['ch_names'][best_idx]} "
                      f"acc={result['real_values'][best_idx]*100:.2f}% "
                      f"p_corrige={result['pvals_corrected'][best_idx]:.4f}")
                print(f"  -> {out_file}")
                print()
            continue

        print(f"=== {args.family_name} x {state} ===")
        result = correct_one_state(results_dir, args.keys, state)
        if result is None:
            continue

        out_file = args.output_path / f"{args.family_name}_{state}_maxstat.npz"
        np.savez_compressed(out_file, **result)

        n_sig = int((result["pvals_corrected"] < 0.05).sum())
        print(f"  {result['n_tests']} tests regroupes, {result['n_perm']} perms")
        print(f"  {n_sig}/{result['n_tests']} significatifs a p<0.05 (corrige)")
        best_idx = result["pvals_corrected"].argmin()
        print(f"  meilleur : {result['test_labels'][best_idx]} "
              f"acc={result['real_values'][best_idx]*100:.2f}% "
              f"p_corrige={result['pvals_corrected'][best_idx]:.4f}")
        print(f"  -> {out_file}")
        print()
