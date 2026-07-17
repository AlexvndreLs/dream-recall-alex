"""Convergence du bootstrap : moyenne cumulée des accuracies sur les 1000 tirages.

Figure de DIAGNOSTIC, pas de résultat : elle vérifie que le nombre de bootstraps
suffit à stabiliser l'estimation, elle ne dit rien sur HR vs LR.

À lire avec réserve : une moyenne cumulée converge mécaniquement en 1/sqrt(n),
donc une courbe plate prouve surtout que l'arithmétique fonctionne. Ce qu'elle
montre d'utile, c'est l'AMPLITUDE résiduelle des oscillations en fin de course :
si elles restent larges devant l'écart entre deux features qu'on veut comparer,
1000 bootstraps ne suffisent pas pour les départager.

La bande grise est l'erreur standard cumulée (std/sqrt(n)) : elle matérialise ce
1/sqrt(n) attendu. Une courbe qui sort de sa propre bande signale un problème
(tirages non indépendants, dérive).

Usage :
    python plot_bootstrap_convergence.py \
        --save-path /scratch/alouis/dream_features_noica_1000hz_overlap \
        --out-dir   /scratch/alouis/dream-recall-alex/plot_overlap \
        --features cosp_sigma/S2 cosp_delta/SWS cov/REM psd_sigma/S2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    RESOLUTION,
    band_label,
    is_matrix_key,
    key_color,
    load_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+",
                   default=["cosp_sigma/S2", "cosp_delta/SWS", "cov/REM", "psd_sigma/S2"],
                   help="Liste 'feature/state'.")
    return p.parse_args()


def cumulative(scores: np.ndarray):
    """Moyenne cumulée et erreur standard cumulée, en %.

    scores : (n_boot,) accuracies, chacune déjà moyennée sur les 324 splits.
    """
    n = np.arange(1, len(scores) + 1)
    cummean = np.cumsum(scores) / n
    # std cumulée par la formule de König-Huygens : évite une boucle sur n_boot
    cumvar = np.cumsum(scores ** 2) / n - cummean ** 2
    cumse = np.sqrt(np.maximum(cumvar, 0) / n)
    return cummean * 100, cumse * 100


def main() -> None:
    args = parse_args()
    print("=== convergence du bootstrap ===")

    pairs = []
    for tok in args.features:
        if "/" not in tok:
            raise SystemExit(f"--features : format attendu feature/state, reçu '{tok}'")
        feat, state = tok.split("/", 1)
        pairs.append((feat.strip(), state.strip()))

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.4),
                             squeeze=False)
    axes = axes[0]

    for ax, (key, state) in zip(axes, pairs):
        d = load_result(args.save_path, key, state)
        if d is None:
            print(f"  absent : {key}_{state}.npz")
            ax.axis("off")
            continue

        scores = np.asarray(d["acc_scores"])
        # Pour une feature vectorielle, acc_scores est (n_boot, 19) : on suit la
        # meilleure électrode, cohérent avec ce que les autres figures testent.
        ch = None
        if not is_matrix_key(key):
            best = int(np.asarray(d["acc_mean"]).argmax())
            ch = str(d["ch_names"][best]) if "ch_names" in d.files else f"élec {best}"
            scores = scores[:, best]

        cummean, cumse = cumulative(scores)
        n = np.arange(1, len(scores) + 1)

        ax.fill_between(n, cummean - cumse, cummean + cumse,
                        color="0.8", lw=0, label="± SE cumulée")
        ax.plot(n, cummean, color=key_color(key), lw=1.5)
        ax.axhline(cummean[-1], color="k", ls="--", lw=0.8)

        title = f"{band_label(key)} — {state}"
        if ch:
            title += f" ({ch})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Bootstraps cumulés")
        ax.set_xscale("log")  # l'essentiel de la convergence se joue avant n=100
        ax.spines[["top", "right"]].set_visible(False)

        # Amplitude résiduelle : le seul chiffre vraiment informatif ici.
        tail = cummean[len(cummean) // 2:]
        ax.text(0.97, 0.06,
                f"final = {cummean[-1]:.2f}%\n"
                f"amplitude 2e moitié = {tail.max() - tail.min():.2f} pt",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.8", alpha=0.85))

    axes[0].set_ylabel("Accuracy moyenne cumulée (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Convergence du bootstrap (diagnostic) — la moyenne cumulée converge "
        "en 1/√n par construction ;\nc'est l'amplitude résiduelle qui indique si "
        "1000 tirages suffisent à départager deux features.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "bootstrap_convergence.png"
    fig.savefig(out, dpi=RESOLUTION)
    plt.close(fig)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
