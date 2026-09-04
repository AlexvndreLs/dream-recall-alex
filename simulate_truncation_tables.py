"""Regenere les deux tableaux simules de la section 1 du rapport.

  tab:regimes  -- comportement de la troncature selon la taille des groupes
  tab:loterie  -- variabilite de la valeur de p a effet constant

Contexte
--------
Le schema de permutation du code publie (_combinations, ttest_perm_indep.py
lignes 303-310) tronque itertools.combinations aux n_perm premiers elements en
ordre lexicographique. Ce n'est pas un tirage, c'est un compteur : il incremente
le dernier indice, puis l'avant-dernier, etc. Aux tailles d'echantillon du
niveau epoque, chaque "permutation" ne deplace donc qu'une ou deux observations,
la nulle reste collee a la statistique observee, et la valeur de p cesse d'etre
une fonction de la taille de l'effet.

Reproductibilite
----------------
Toutes les graines sont fixees. Les deux tableaux sortent a l'identique a chaque
execution. Verifie le 2026-09-04, les 6 valeurs de tab:loterie et les 35 valeurs
de tab:regimes coincident avec celles imprimees dans le rapport.

Cout
----
Quelques minutes en tout, sur un seul coeur. Aucune dependance au cluster ni aux
donnees, tout est synthetique. Ne pas soumettre en batch, lancer directement.

Usage
-----
    python simulate_truncation_tables.py            # les deux tableaux
    python simulate_truncation_tables.py --only regimes
    python simulate_truncation_tables.py --only loterie --csv loterie.csv
"""

import argparse
from itertools import combinations, islice

import numpy as np
from scipy.stats import ttest_ind


# ---------------------------------------------------------------------------
# tab:regimes : un seul canal, deux nulles comparees
# ---------------------------------------------------------------------------
def regime(k, n_perm=200, effect=1.0, seed=0, seed_random_null=1):
    """Un canal, k observations par groupe, effet reel de 1 ecart-type.

    Compare la nulle tronquee (schema publie) a une vraie nulle aleatoire.
    Retourne (t_obs, moy_tronq, std_tronq, p_tronq, moy_alea, std_alea, p_alea).
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(k) + effect
    b = rng.randn(k)
    full = np.concatenate([a, b])
    n = 2 * k
    t_obs = ttest_ind(a, b, equal_var=False)[0]

    # nulle tronquee : les n_perm premieres combinaisons, ordre lexicographique
    lex = []
    for comb in islice(combinations(range(n), k), n_perm):
        comb = list(comb)
        rest = [i for i in range(n) if i not in set(comb)]
        lex.append(ttest_ind(full[comb], full[rest], equal_var=False)[0])
    lex = np.array(lex[1:])  # on retire la combinaison identite

    # nulle aleatoire : le test de permutation correct
    rr = np.random.RandomState(seed_random_null)
    rnd = []
    for _ in range(n_perm):
        idx = rr.choice(n, k, replace=False)
        rest = np.setdiff1d(np.arange(n), idx)
        rnd.append(ttest_ind(full[idx], full[rest], equal_var=False)[0])
    rnd = np.array(rnd)

    return (t_obs,
            lex.mean(), lex.std(), (np.abs(lex) >= abs(t_obs)).mean(),
            rnd.mean(), rnd.std(), (np.abs(rnd) >= abs(t_obs)).mean())


def table_regimes(sizes=(5, 20, 100, 500, 2000)):
    print("=== tab:regimes ===")
    print("un seul canal, effet reel de 1 ecart-type, 200 permutations\n")
    print(f"{'n':>5} | {'t_obs':>7} || {'moy':>7} {'std':>5} {'p':>6}"
          f" || {'moy':>7} {'std':>5} {'p':>6}")
    print(f"{'':>5} | {'':>7} || {'nulle tronquee':^21}"
          f" || {'nulle aleatoire':^21}")
    for k in sizes:
        t, m1, s1, p1, m2, s2, p2 = regime(k)
        print(f"{k:5d} | {t:+7.2f} || {m1:+7.2f} {s1:5.2f} {p1:6.3f}"
              f" || {m2:+7.2f} {s2:5.2f} {p2:6.3f}")
    print()


# ---------------------------------------------------------------------------
# tab:loterie : 19 canaux, correction maxstat, 15 jeux de donnees
# ---------------------------------------------------------------------------
def one_dataset(seed, k=1000, n_perm=200, n_elec=19, effect=1.0):
    """19 canaux, effet sur l'electrode 0 seulement, correction maxstat.

    Retourne (max |t| observe, p minimale sur les 19, nb d'electrodes sous 0.05).
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(k, n_elec)
    b = rng.randn(k, n_elec)
    a[:, 0] += effect
    full = np.vstack([a, b])
    n = 2 * k
    t_obs = ttest_ind(a, b, equal_var=False)[0]

    lex = []
    for comb in islice(combinations(range(n), k), n_perm):
        comb = list(comb)
        rest = [i for i in range(n) if i not in set(comb)]
        lex.append(ttest_ind(full[comb], full[rest], equal_var=False)[0])
    lex = np.array(lex[1:])

    null_max = np.abs(lex).max(axis=1)
    pvals = np.array([(null_max >= abs(t)).mean() for t in t_obs])
    return np.abs(t_obs).max(), pvals.min(), int((pvals < 0.05).sum())


def table_loterie(n_datasets=15, csv=None):
    print("=== tab:loterie ===")
    print("1000 observations par groupe, 19 electrodes, effet identique partout,")
    print("200 permutations, correction par statistique du maximum\n")
    print(f"{'graine':>6} | {'max|t_obs|':>10} | {'p_min':>6} | nb elec p<0.05")
    rows = []
    for seed in range(n_datasets):
        t_max, p_min, n_sig = one_dataset(seed)
        rows.append((seed, t_max, p_min, n_sig))
        print(f"{seed:6d} | {t_max:10.2f} | {p_min:6.3f} | {n_sig}/19")

    t = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    print()
    print("ATTENTION : chaque colonne est ordonnee separement. La ligne")
    print("'minimum' ne correspond pas a un jeu de donnees unique.")
    print(f"  minimum sur {n_datasets} tirages : {t.min():.2f} & {p.min():.3f}")
    print(f"  mediane                   : {np.median(t):.2f} & {np.median(p):.3f}")
    print(f"  maximum sur {n_datasets} tirages : {t.max():.2f} & {p.max():.3f}")
    print(f"  tirages sous p = 0.05     : {(p < 0.05).sum()}/{n_datasets}")

    if csv:
        with open(csv, "w", encoding="utf-8") as f:
            f.write("seed,max_abs_t_obs,p_min,n_elec_sig\n")
            for s, tm, pm, ns in rows:
                f.write(f"{s},{tm:.4f},{pm:.4f},{ns}\n")
        print(f"\ncsv : {csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["regimes", "loterie"], default=None)
    ap.add_argument("--n-datasets", type=int, default=15)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if args.only in (None, "regimes"):
        table_regimes()
    if args.only in (None, "loterie"):
        table_loterie(args.n_datasets, args.csv)


if __name__ == "__main__":
    main()
