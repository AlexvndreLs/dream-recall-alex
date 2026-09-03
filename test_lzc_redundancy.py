"""Protocole de redondance en deux etapes, section 3.3 du rapport.

Repond a la question qui conditionne l'interpretation de toute la section
complexite : une mesure de complexite apporte-t-elle une information que
l'exposant aperiodique ne porte pas deja ? La pente spectrale est en effet
elle-meme une mesure indirecte de regularite temporelle, et une mesure de
complexite qui lui est fortement correlee ne mesure rien de nouveau.

Etape 1, correlation de Spearman par etat et par electrode entre la mesure de
complexite et l'exposant aperiodique. Le rapport pose |rho| > 0.8 comme seuil
de redondance. Trois niveaux d'agregation sont calcules, parce qu'ils ne
disent pas la meme chose et que confondre les deux premiers est l'erreur
classique :

  rho_within   correlation calculee A L'INTERIEUR de chaque sujet sur ses
               propres epoques, puis moyennee entre sujets par transformation
               de Fisher. Mesure le couplage epoque par epoque des deux
               quantites, independamment des differences entre sujets.
  rho_pooled   correlation sur toutes les epoques de tous les sujets
               melangees. Melange variance intra et inter sujet, donc gonflee
               par toute difference de niveau entre sujets. Rapportee ici pour
               montrer l'ecart avec rho_within, pas comme mesure de reference.
  rho_submean  correlation entre les moyennes par sujet, 36 points par
               electrode. C'est le niveau auquel se joue reellement le
               contraste HR contre LR, donc le plus pertinent pour savoir si
               la complexite peut discriminer la ou l'exposant echoue.

Etape 2, gain marginal de decodage. Accuracy d'une LDA sur l'exposant seul,
comparee a celle d'une LDA sur l'exposant et la mesure de complexite
ensemble, electrode par electrode. Le rapport pose un gain inferieur a 1 ou
2 points comme signe que la complexite n'ajoute rien de decodable.

Les deux etapes sont necessaires. Une correlation moderee n'exclut pas que la
part non partagee soit du bruit, et c'est le gain marginal qui le dit.
Inversement une correlation forte suffit a conclure sans aller plus loin.

Machinerie
----------
Tout est importe de classify.py : bootstrap_sample, StratifiedLeave2GroupsOut,
run_cv, _seed, permute_subject_labels, compute_global_n_trials. Les splits,
les graines et le n_trials sont donc rigoureusement les memes que ceux des
accuracies deja publiees dans le tableau, et les deux modeles de l'etape 2
sont evalues sur le MEME echantillon bootstrap et les MEMES splits, ce qui
rend leur difference appariee.

LDA est invariante par transformation lineaire inversible des features, donc
melanger deux features d'echelles tres differentes (exposant autour de 1 a 3,
LZC autour de 0.3 a 0.7) est sans consequence mathematique. --normalize
ajoute un StandardScaler ajuste sur le train uniquement, disponible si on
veut s'en assurer numeriquement, off par defaut comme dans classify.py.

Sur --n-bootstraps 200 par defaut, contre 1000 dans classify.py : le gain est
une difference appariee, sa variance est bien plus faible que celle de chaque
accuracy prise isolement, et 200 tirages suffisent largement. Le cout de
l'etape 2 est de deux classifications vectorielles completes, donc environ
deux fois celui d'un job batch_classify_vector.

Sur les permutations, --n-perm 0 par defaut : l'intervalle bootstrap sur le
gain suffit pour le critere du rapport, qui est descriptif. --n-perm N
construit en plus une loi nulle du gain par permutation des labels au niveau
sujet, avec correction max-stat sur les 19 electrodes, au prix de N fois le
cout de l'etape 2. Ne l'activer que si on veut une valeur de p publiable sur
le gain lui-meme.

Usage
-----
    # etape 1 seule, quelques secondes
    python test_lzc_redundancy.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz \
        --out-dir   /home/alouis/dream-recall-alex/plot_noverlap_lzc \
        --step 1

    # les deux etapes
    python test_lzc_redundancy.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz \
        --out-dir   /home/alouis/dream-recall-alex/plot_noverlap_lzc \
        --step both --n-jobs $SLURM_CPUS_PER_TASK --n-bootstraps 200

    # applicable a n'importe quelle mesure de complexite, pas seulement la LZC
    python test_lzc_redundancy.py ... --key-complexity higuchi_fd
"""

import argparse
from pathlib import Path
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config_v3 import CH_NAMES, N_EEG, STATE_LIST, SUBJECT_LABELS, SUBJECT_LIST_ORDERED
from classify import (
    PERM_SEED_OFFSET,
    StratifiedLeave2GroupsOut,
    _seed,
    bootstrap_sample,
    compute_global_n_trials,
    load_subject,
    permute_subject_labels,
    run_cv,
)

ELEC = CH_NAMES[:N_EEG]
REDUNDANT_RHO = 0.8   # seuil de redondance pose par le rapport
GAIN_REF = (1.0, 2.0)  # points d'accuracy, seuils indicatifs du rapport


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--key-complexity", type=str, default="lzc")
    p.add_argument("--key-reference", type=str, default="aperiodic",
                   help="Mesure contre laquelle tester la redondance.")
    p.add_argument("--states", nargs="+", default=STATE_LIST)
    p.add_argument("--step", choices=["1", "2", "both"], default="both")
    p.add_argument("--n-bootstraps", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=0,
                   help="Loi nulle du gain par permutation sujet. 0 = desactive.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--normalize", action="store_true", default=False)
    p.add_argument("--force-n-trials", type=int, default=None)
    return p.parse_args()


# --- chargement apparie -----------------------------------------------------

def load_pair(save_path: Path, key_a: str, key_b: str, state: str):
    """Charge deux features scalaires alignees epoque par epoque.

    Les deux cles proviennent de la meme segmentation
    (load_epochs_by_atomic_stage), donc l'ordre des epoques est identique. On
    le verifie quand meme par les shapes : une divergence signalerait une
    extraction faite sur un autre decoupage, ce qui invaliderait toute
    correlation appariee.
    """
    a_list, b_list, labels, subs = [], [], [], []
    for sub_id, label in zip(SUBJECT_LIST_ORDERED, SUBJECT_LABELS):
        a = load_subject(save_path, key_a, sub_id, state)
        b = load_subject(save_path, key_b, sub_id, state)
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            raise RuntimeError(
                f"sub-{sub_id} {state} : {key_a} {a.shape} contre {key_b} "
                f"{b.shape}. Les deux features ne viennent pas du meme "
                f"decoupage en epoques, comparaison appariee impossible."
            )
        a_list.append(a)
        b_list.append(b)
        labels.append(label)
        subs.append(sub_id)
    return a_list, b_list, np.array(labels), subs


# --- etape 1 : correlation de Spearman --------------------------------------

def _spearman_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, p) x (n, p) -> (p,) rho de Spearman colonne par colonne.

    Rangs le long de l'axe des echantillons, puis correlation de Pearson sur
    les rangs, ce qui est la definition de Spearman. Vectorise, contrairement
    a un appel scipy par colonne.
    """
    rx = rankdata(x, axis=0)
    ry = rankdata(y, axis=0)
    rx = rx - rx.mean(axis=0)
    ry = ry - ry.mean(axis=0)
    denom = np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (rx * ry).sum(axis=0) / denom, np.nan)


def _fisher_mean(rhos: np.ndarray) -> np.ndarray:
    """Moyenne de correlations via la transformation z de Fisher.

    Moyenner des rho bruts sous-estime la correlation moyenne, la relation
    n'etant pas lineaire. On passe par z = arctanh(rho), on moyenne, on
    revient. Les rho a +/-1 exact sont ecartes, arctanh y diverge.
    """
    z = np.arctanh(np.clip(rhos, -0.999999, 0.999999))
    return np.tanh(np.nanmean(z, axis=0))


def step1(save_path: Path, key_c: str, key_r: str, states: list) -> pd.DataFrame:
    rows = []
    for state in states:
        c_list, r_list, _, subs = load_pair(save_path, key_c, key_r, state)
        if not c_list:
            print(f"  {state} : aucun sujet avec les deux cles, skip")
            continue

        # intra sujet, puis moyenne de Fisher
        per_sub = np.array([
            _spearman_columns(c, r) for c, r in zip(c_list, r_list)
        ])                                    # (n_sub, 19)
        rho_within = _fisher_mean(per_sub)
        rho_within_sd = np.nanstd(per_sub, axis=0)

        # toutes epoques melangees
        rho_pooled = _spearman_columns(
            np.concatenate(c_list, axis=0), np.concatenate(r_list, axis=0)
        )

        # moyennes par sujet
        rho_submean = _spearman_columns(
            np.array([c.mean(axis=0) for c in c_list]),
            np.array([r.mean(axis=0) for r in r_list]),
        )

        for i, ch in enumerate(ELEC):
            rows.append(dict(
                state=state, electrode=ch,
                rho_within=rho_within[i], rho_within_sd=rho_within_sd[i],
                rho_pooled=rho_pooled[i], rho_submean=rho_submean[i],
                n_subjects=len(subs),
            ))
        print(f"  {state} : n={len(subs)} sujets  |  "
              f"|rho_within| med={np.nanmedian(np.abs(rho_within)):.3f} "
              f"max={np.nanmax(np.abs(rho_within)):.3f}  |  "
              f"|rho_submean| max={np.nanmax(np.abs(rho_submean)):.3f}")
    return pd.DataFrame(rows)


def plot_step1(df: pd.DataFrame, out_dir: Path, key_c: str, key_r: str) -> None:
    states = list(dict.fromkeys(df["state"]))
    panels = [("rho_within", "Within subject (Fisher mean)"),
              ("rho_submean", "Between subjects (subject means)")]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.2 * len(panels) + 1.6))
    axes = np.atleast_1d(axes)
    for ax, (col, title) in zip(axes, panels):
        mat = np.array([
            df[df["state"] == s].set_index("electrode").loc[ELEC, col].values
            for s in states
        ])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ELEC)))
        ax.set_xticklabels(ELEC, rotation=90, fontsize=7)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels(states, fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v) and abs(v) >= REDUNDANT_RHO:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=11, color="white")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Spearman rho")

    fig.suptitle(f"Redundancy step 1: {key_c} versus {key_r}. "
                 f"Star marks |rho| >= {REDUNDANT_RHO}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = out_dir / f"redundancy_step1_{key_c}_vs_{key_r}.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    print(f"  figure : {f}")


# --- etape 2 : gain marginal de decodage ------------------------------------

def _make_clf(normalize: bool):
    return (Pipeline([("scaler", StandardScaler()), ("lda", LDA(solver="svd"))])
            if normalize else LDA(solver="svd"))


def _one_draw(clf, cv, data, labels, n_trials, key, state, i,
              perm: bool, n_perm: int) -> np.ndarray:
    """Un tirage -> (2, 19) accuracies : modele reference, modele reference+complexite.

    data est une liste de (n_epochs, 19, 2), derniere dimension
    [reference, complexite]. Les deux modeles voient le MEME echantillon et
    les MEMES splits : leur difference est donc appariee.
    """
    if perm:
        labels = permute_subject_labels(
            labels, _seed("perm", state, PERM_SEED_OFFSET + n_perm + i)
        )
        seed = _seed("perm", state, PERM_SEED_OFFSET + i)
    else:
        seed = _seed(key, state, i)

    X, y, groups = bootstrap_sample(data, labels, n_trials, seed)
    splits = list(cv.split(X, y, groups))

    out = np.empty((2, X.shape[1]))
    for e in range(X.shape[1]):
        out[0, e] = run_cv(clf, splits, X[:, e, 0:1], y)   # reference seule
        out[1, e] = run_cv(clf, splits, X[:, e, :],   y)   # reference + complexite
    return out


def step2(save_path: Path, key_c: str, key_r: str, states: list,
          n_trials: int, n_bootstraps: int, n_perm: int,
          n_jobs: int, normalize: bool) -> pd.DataFrame:
    clf, cv = _make_clf(normalize), StratifiedLeave2GroupsOut()
    rows = []

    for state in states:
        c_list, r_list, labels, subs = load_pair(save_path, key_c, key_r, state)
        if len(c_list) < 4:
            print(f"  {state} : {len(c_list)} sujets, cohorte insuffisante, skip")
            continue
        # (n_epochs, 19, 2), ordre [reference, complexite]
        data = [np.stack([r, c], axis=-1) for c, r in zip(c_list, r_list)]

        t0 = time()
        draws = np.array(Parallel(n_jobs=n_jobs)(
            delayed(_one_draw)(clf, cv, data, labels, n_trials,
                               key_c, state, i, False, n_perm)
            for i in range(n_bootstraps)
        ))                                    # (n_boot, 2, 19)
        acc_ref = draws[:, 0, :].mean(axis=0)
        acc_both = draws[:, 1, :].mean(axis=0)
        gain = (draws[:, 1, :] - draws[:, 0, :]) * 100   # points, apparie
        gain_mean = gain.mean(axis=0)
        gain_lo = np.percentile(gain, 2.5, axis=0)
        gain_hi = np.percentile(gain, 97.5, axis=0)

        pvals = np.full(len(ELEC), np.nan)
        if n_perm > 0:
            perms = np.array(Parallel(n_jobs=n_jobs)(
                delayed(_one_draw)(clf, cv, data, labels, n_trials,
                                   key_c, state, p, True, n_perm)
                for p in range(n_perm)
            ))
            null_gain = (perms[:, 1, :] - perms[:, 0, :]) * 100   # (n_perm, 19)
            # max-stat sur les 19 electrodes, meme convention que
            # compute_maxstat_correction.py en mode arthur
            null_max = null_gain.max(axis=1)
            pvals = (np.sum(null_max[:, None] >= gain_mean[None, :], axis=0) + 1) \
                    / (n_perm + 1)

        for i, ch in enumerate(ELEC):
            rows.append(dict(
                state=state, electrode=ch,
                acc_reference_pct=acc_ref[i] * 100,
                acc_both_pct=acc_both[i] * 100,
                gain_pts=gain_mean[i],
                gain_ci_lo=gain_lo[i], gain_ci_hi=gain_hi[i],
                p_gain_maxstat=pvals[i],
                n_subjects=len(subs), n_trials=n_trials,
                n_bootstraps=n_bootstraps, n_perm=n_perm,
            ))
        print(f"  {state} : gain max {gain_mean.max():+.2f} pts sur "
              f"{ELEC[int(np.argmax(gain_mean))]}, median "
              f"{np.median(gain_mean):+.2f} pts, {time() - t0:.0f}s")
    return pd.DataFrame(rows)


def plot_step2(df: pd.DataFrame, out_dir: Path, key_c: str, key_r: str) -> None:
    states = list(dict.fromkeys(df["state"]))
    fig, axes = plt.subplots(len(states), 1, sharex=True,
                             figsize=(11, 1.9 * len(states) + 1.4))
    axes = np.atleast_1d(axes)
    for ax, state in zip(axes, states):
        sub = df[df["state"] == state].set_index("electrode").loc[ELEC]
        x = np.arange(len(ELEC))
        ax.bar(x, sub["gain_pts"], color="#4C72B0", width=0.75)
        ax.errorbar(x, sub["gain_pts"],
                    yerr=[sub["gain_pts"] - sub["gain_ci_lo"],
                          sub["gain_ci_hi"] - sub["gain_pts"]],
                    fmt="none", ecolor="0.3", elinewidth=0.8, capsize=2)
        for ref in GAIN_REF:
            ax.axhline(ref, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(state, fontsize=9)
    axes[-1].set_xticks(np.arange(len(ELEC)))
    axes[-1].set_xticklabels(ELEC, rotation=90, fontsize=7)
    fig.suptitle(
        f"Redundancy step 2: marginal decoding gain of adding {key_c} "
        f"to {key_r}. Dashed lines at {GAIN_REF[0]:.0f} and {GAIN_REF[1]:.0f} "
        f"accuracy points. Bars show 95 percent bootstrap interval",
        fontsize=9,
    )
    fig.supylabel("Accuracy gain (points)", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    f = out_dir / f"redundancy_step2_{key_c}_vs_{key_r}.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    print(f"  figure : {f}")


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kc, kr = args.key_complexity, args.key_reference
    t0 = time()

    if args.step in ("1", "both"):
        print(f"=== etape 1 : Spearman {kc} contre {kr} ===")
        df1 = step1(args.save_path, kc, kr, args.states)
        if not df1.empty:
            f = args.out_dir / f"redundancy_step1_{kc}_vs_{kr}.csv"
            df1.to_csv(f, index=False)
            print(f"  csv : {f}")
            plot_step1(df1, args.out_dir, kc, kr)
            worst = df1.loc[df1["rho_within"].abs().idxmax()]
            print(f"  verdict etape 1 : |rho_within| max = "
                  f"{abs(worst['rho_within']):.3f} ({worst['state']}/"
                  f"{worst['electrode']}), seuil {REDUNDANT_RHO}")

    if args.step in ("2", "both"):
        n_trials = args.force_n_trials or compute_global_n_trials(
            args.save_path, skip_check=True
        )
        print(f"=== etape 2 : gain marginal, n_trials = {n_trials}, "
              f"{args.n_bootstraps} bootstraps, {args.n_perm} permutations ===")
        df2 = step2(args.save_path, kc, kr, args.states, n_trials,
                    args.n_bootstraps, args.n_perm, args.n_jobs, args.normalize)
        if not df2.empty:
            f = args.out_dir / f"redundancy_step2_{kc}_vs_{kr}.csv"
            df2.to_csv(f, index=False)
            print(f"  csv : {f}")
            plot_step2(df2, args.out_dir, kc, kr)
            best = df2.loc[df2["gain_pts"].idxmax()]
            print(f"  verdict etape 2 : gain max = {best['gain_pts']:+.2f} pts "
                  f"({best['state']}/{best['electrode']}), seuils indicatifs "
                  f"{GAIN_REF[0]:.0f} a {GAIN_REF[1]:.0f} pts")

    m, s = divmod(int(time() - t0), 60)
    print(f"total : {m}m{s:02d}s")