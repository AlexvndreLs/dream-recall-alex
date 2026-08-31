#!/bin/bash
# ==========================================================================
# Remet sur la branche SANS recouvrement les elements de la section 1 qui
# venaient de la branche avec recouvrement, et audite la provenance des
# tableaux.
#
# Concerne :
#   1.1  bootstrap_convergence, bootstrap_dispersion_matrix,
#        les deux constructions de loi nulle
#   1.5  courbes de survie des p-values
#   1.8  verification que fig3_corrigee_S2.png est bien non-overlap
#   tableaux  regeneration du CSV non-overlap + diff contre l'overlap
#
# NE concerne PAS la section 1.9 (EFS) : aucun resultat EFS n'existe sur la
# branche non-overlap, il faudrait relancer batch_classify_efs.sh. Le script
# le signale mais ne lance rien.
#
# Rien n'est ecrase : les figures partent dans plot_noverlap/, le CSV est
# ecrit dans le results/ de sa propre branche (comportement par defaut du
# script, lecture seule sur les .npz).
#
# Usage :
#   cd ~/dream-recall-alex
#   bash fix_section1_branch.sh audit     # n'ecrit rien, diagnostique
#   bash fix_section1_branch.sh run       # audit puis regeneration
# ==========================================================================

set -uo pipefail   # pas de -e : on veut continuer meme si un element manque

MODE="${1:-audit}"
if [[ "$MODE" != "audit" && "$MODE" != "run" ]]; then
    echo "Usage : bash $0 [audit|run]"
    exit 1
fi

cd ~/dream-recall-alex
source /home/alouis/mne_env/bin/activate

NOVL=/scratch/alouis/dream_features_noica_1000hz            # sans recouvrement
OVL=/scratch/alouis/dream_features_noica_1000hz_overlap     # avec recouvrement
OUT=~/dream-recall-alex/plot_noverlap_$(date +%Y%m%d_%H%M)
FIGS=~/dream-recall-alex/figures

CSV_NOVL="$NOVL/results/pvalue_summary_table.csv"
CSV_OVL="$OVL/results/pvalue_summary_table.csv"

hr () { printf '%s\n' "----------------------------------------------------------"; }

# ==========================================================================
echo "##########  PHASE 1, AUDIT"
hr

echo "[a] Les deux CSV de synthese"
for f in "$CSV_NOVL" "$CSV_OVL"; do
    if [ -f "$f" ]; then
        printf "    %s  %8s lignes  %s\n" \
            "$(date -r "$f" '+%Y-%m-%d %H:%M')" "$(wc -l < "$f")" "$f"
    else
        echo "    ABSENT : $f"
    fi
done

hr
echo "[b] Horodatage des figures 1.1 et 1.5, pour retrouver le CSV source"
for f in plot_overlap/fig5_courbe_survie_pvalues.png \
         plot_overlap/bootstrap_convergence.png \
         plot_overlap/bootstrap_dispersion_matrix.png; do
    [ -f "$f" ] && printf "    %s  %s\n" \
        "$(date -r "$f" '+%Y-%m-%d %H:%M')" "$f" || echo "    ABSENT : $f"
done
echo
echo "    Le CSV dont l'horodatage precede immediatement celui de la courbe"
echo "    de survie est celui qui l'a nourrie."

hr
echo "[c] Section 1.8, de quoi fig3_corrigee_S2.png est-il la copie ?"
if [ -f "$FIGS/fig3_corrigee_S2.png" ]; then
    FOUND=""
    for cand in "$FIGS"/fig3_arthur_S2_*.png "$FIGS"/fig3_ffx_vs_rfx_S2*.png; do
        [ -f "$cand" ] || continue
        if cmp -s "$FIGS/fig3_corrigee_S2.png" "$cand"; then
            echo "    identique a : $(basename "$cand")"
            FOUND="$cand"
        fi
    done
    if [ -z "$FOUND" ]; then
        echo "    aucun jumeau trouve, empreintes de toute la serie :"
        md5sum "$FIGS"/fig3_*S2*.png | sed 's/^/      /'
    else
        case "$(basename "$FOUND")" in
            *arthurfull*|*correct*|*arthurbug*|*arthurRFX*)
                echo "    -> produit par batch_fig3_arthurfull.sh ou"
                echo "       batch_fig3_ttest_compare.sh, qui tournent tous"
                echo "       deux sur $NOVL. Section 1.8 est NON-OVERLAP, ok." ;;
            *_overlap*)
                echo "    -> produit par batch_fig3_ffx_rfx_overlap.sh, donc"
                echo "       branche AVEC recouvrement. A corriger." ;;
            *)
                echo "    -> provenance a determiner a la main." ;;
        esac
    fi
else
    echo "    ABSENT : $FIGS/fig3_corrigee_S2.png"
fi

hr
echo "[d] Section 1.9, resultats EFS sur la branche non-overlap ?"
N_EFS=$(ls "$NOVL"/results/*efs* 2>/dev/null | wc -l)
echo "    $N_EFS fichier(s) EFS dans $NOVL/results"
if [ "$N_EFS" -eq 0 ]; then
    echo "    -> fig08 restera sur la branche avec recouvrement."
    echo "       Rien a lancer ici, c'est un recalcul SLURM complet."
    echo "       A assumer dans la legende de la figure."
fi

hr
echo "[e] Prerequis des deux constructions de loi nulle (1.1)"
for suf in _epochperm _epochperm_fixed; do
    N=$(ls "$NOVL"/results/*"${suf}".npz 2>/dev/null | wc -l)
    printf "    %-20s %2d / 24 dans %s/results\n" "*${suf}.npz" "$N" "$NOVL"
done

if [ "$MODE" = "audit" ]; then
    echo
    echo "Audit termine, rien n'a ete ecrit. Relancer avec : bash $0 run"
    exit 0
fi

# ==========================================================================
echo
echo "##########  PHASE 2, REGENERATION SUR LA BRANCHE NON-OVERLAP"
hr
if [ -e "$OUT" ]; then
    echo "ARRET : $OUT existe deja, refus d'ecraser. Renomme-le ou attends"
    echo "une minute, le nom du dossier est horodate a la minute."
    exit 1
fi
mkdir -p "$OUT"
echo "Sortie : $OUT  (dossier neuf, rien d'existant n'est touche)"

echo "[1] Tableaux, controle du CSV non-overlap existant"
echo "    AUCUNE reconstruction : le CSV existe deja, on ne fait que le lire."
if [ -f "$CSV_NOVL" ]; then
    N_PENDING=$(grep -c "PENDING" "$CSV_NOVL" 2>/dev/null || echo 0)
    N_NA=$(grep -c "N/A" "$CSV_NOVL" 2>/dev/null || echo 0)
    echo "    $N_PENDING ligne(s) PENDING, $N_NA ligne(s) N/A"
    if [ "$N_PENDING" -gt 0 ]; then
        echo "    ATTENTION : des colonnes epoch etaient encore en attente"
        echo "    quand ce CSV a ete ecrit. Les tableaux qui en derivent sont"
        echo "    incomplets. Pour le refaire, explicitement et a la main :"
        echo "      cp $CSV_NOVL ${CSV_NOVL%.csv}_$(date +%Y%m%d).csv"
        echo "      python build_pvalue_summary_table.py --save-path $NOVL"
    else
        echo "    -> CSV complet, utilisable tel quel."
    fi
else
    echo "    ABSENT, les courbes de survie ne pourront pas etre tracees."
fi

hr
echo "[2] Diff des tableaux entre les deux branches"
if [ -f "$CSV_NOVL" ] && [ -f "$CSV_OVL" ]; then
    python - "$CSV_NOVL" "$CSV_OVL" <<'PY'
import sys
import pandas as pd

novl = pd.read_csv(sys.argv[1])
ovl = pd.read_csv(sys.argv[2])

keys = [c for c in ("feature", "key", "state") if c in novl.columns]
if not keys:
    print("    colonnes de jointure introuvables, colonnes disponibles :")
    print("   ", list(novl.columns))
    raise SystemExit

m = novl.merge(ovl, on=keys, suffixes=("_novl", "_ovl"))
print(f"    {len(m)} combinaisons appariees sur {keys}")

num = [c for c in novl.columns
       if c not in keys and pd.api.types.is_numeric_dtype(novl[c])]

print(f"\n    {'colonne':<34} {'ecart median':>13} {'ecart max':>11}")
print("    " + "-" * 60)
for c in num:
    a = pd.to_numeric(m[f"{c}_novl"], errors="coerce")
    b = pd.to_numeric(m[f"{c}_ovl"], errors="coerce")
    d = (a - b).abs().dropna()
    if len(d):
        print(f"    {c:<34} {d.median():13.4f} {d.max():11.4f}")

# Ce qui compte vraiment : les combinaisons qui changent de statut.
for col in [c for c in num if c.startswith("p_")]:
    a = pd.to_numeric(m[f"{col}_novl"], errors="coerce")
    b = pd.to_numeric(m[f"{col}_ovl"], errors="coerce")
    flip = m[((a < 0.05) & (b >= 0.05)) | ((a >= 0.05) & (b < 0.05))]
    if len(flip):
        print(f"\n    {col} : {len(flip)} combinaison(s) changent de statut a 0.05")
        print(flip[keys].to_string(index=False, max_rows=15))
PY
else
    echo "    diff impossible, un des deux CSV manque"
fi

hr
echo "[3] Section 1.1, bootstrap"
python plot_bootstrap_convergence.py \
    --save-path "$NOVL" --out-dir "$OUT" \
    --features cosp_sigma/S2 cosp_delta/SWS cov/REM psd_sigma/S2

python plot_bootstrap_dispersion.py \
    --save-path "$NOVL" --out-dir "$OUT" --family matrix

hr
echo "[4] Section 1.1, les deux constructions de loi nulle"
N_EP=$(ls "$NOVL"/results/*_epochperm.npz 2>/dev/null | wc -l)
N_FX=$(ls "$NOVL"/results/*_epochperm_fixed.npz 2>/dev/null | wc -l)

if [ "$N_EP" -ge 24 ]; then
    python plot_barplot_riemann_epoch_corr.py \
        --save-path "$NOVL" --out-dir "$OUT" --alpha 0.001 --correction global
else
    echo "    SAUTE, seulement $N_EP/24 *_epochperm.npz"
    echo "    -> sbatch batch_recompute_perms_epoch_matrix.sh"
fi

if [ "$N_FX" -ge 24 ]; then
    python plot_barplot_riemann_ffx_fixed.py \
        --save-path "$NOVL" --out-dir "$OUT" --alpha 0.001 --correction global
else
    echo "    SAUTE, seulement $N_FX/24 *_epochperm_fixed.npz"
    echo "    -> sbatch batch_recompute_perms_epoch_fixed_matrix.sh"
fi

hr
echo "[5] Section 1.5, courbes de survie"
if [ -f "$CSV_NOVL" ]; then
    python plot_correction_summary.py \
        --csv "$CSV_NOVL" --out-dir "$OUT" --alpha 0.05
else
    echo "    SAUTE, $CSV_NOVL absent"
fi

hr
echo "[6] Recopie vers final_plotted_figures, SANS ecrasement"
FINAL=~/dream-recall-alex/final_plotted_figures
mkdir -p "$FINAL"

# La copie n'a lieu que si la destination n'existe pas : le test [ -e ]
# ci-dessous garantit qu'aucun fichier existant n'est touche. Un fichier deja
# present et different est conserve tel quel et signale, a toi de decider si
# tu veux le remplacer par la version non-overlap.
copy_new () {
    # copy_new <source> <nom de destination>
    local src="$1" dst="$FINAL/$2"
    [ -f "$src" ] || { echo "    absent, saute : $(basename "$src")"; return; }
    if [ -e "$dst" ]; then
        if cmp -s "$src" "$dst"; then
            echo "    identique deja en place : $2"
        else
            echo "    CONSERVE l'existant, differe de la nouvelle version : $2"
            echo "        ancien : $dst"
            echo "        nouveau: $src"
        fi
    else
        cp "$src" "$dst" && echo "    copie : $2"
    fi
}

copy_new "$OUT/bootstrap_convergence.png"                          "fig01a_bootstrap_convergence.png"
copy_new "$OUT/bootstrap_dispersion_matrix.png"                    "fig01b_bootstrap_dispersion.png"
copy_new "$OUT/fig5_courbe_survie_pvalues.png"                     "fig06_survie_pvalues.png"
copy_new "$OUT/barplot_riemann_epoch_maxstat_global_p0.001.png"    "figA1_nulle_rebootstrap.png"
copy_new "$OUT/barplot_riemann_epoch_fixed_maxstat_global_p0.001.png" "figA2_nulle_echantillon_fige.png"

hr
echo "##########  RESULTAT"
echo "Dossier de travail, tout ce qui a ete produit :"
ls -la "$OUT"
echo
echo "final_plotted_figures :"
ls -la "$FINAL"
echo
echo "Ces fichiers sont sur la branche SANS recouvrement."
echo "Restent sur la branche AVEC recouvrement, assume :"
echo "  fig08_efs_regions  section 1.9, recalcul SLURM non lance"
echo
echo "Aucun fichier n'a ete supprime ni ecrase. Les figures deja presentes"
echo "dans final_plotted_figures qui different de la nouvelle version sont"
echo "listees ci-dessus, le remplacement est a faire a la main."