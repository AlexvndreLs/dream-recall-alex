# dream-recall-alex

Classification High vs Low dream recallers (HR/LR) à partir d'EEG de sommeil.
Réplication et extension du chapitre 1 de la thèse d'A. Dehgan
([arthurdehgan/sleep](https://github.com/arthurdehgan/sleep)).

## Installation

Python >= 3.11 requis.

```bash
pip install --pre -e .
```

Le flag `--pre` est nécessaire : `specparam` (successeur de FOOOF) ne publie
que des release candidates, donc `specparam==2.0.0rc7` n'est pas installable
sans lui.

Les versions des dépendances sont épinglées à l'identique de l'environnement
`mne_env` utilisé sur le cluster. Les résultats dépendent du comportement
exact de specparam, pyriemann et MNE.

## Pipeline

Quatre étapes, à exécuter dans l'ordre. Chacune lit ce que la précédente a
écrit.

### 1. MATLAB brut vers BIDS

```bash
dream-bids 5 --data-path /chemin/vers/sleep_raw_data \
             --bids-path /chemin/vers/dream_bids
```

Lit les enregistrements `.mat` et leurs hypnogrammes, écrit un dataset BIDS
(25 canaux, 1000 Hz, référence nez).

### 2. Prétraitement

```bash
dream-preprocess 5 --bids-path /chemin/vers/dream_bids \
                   --deriv-root /chemin/vers/dream_bids/derivatives \
                   --branches noica ica iclabel
```

Écrit un derivative BIDS par branche sous `derivatives/preprocessed-<branche>/`.

### 3. Extraction des features

```bash
dream-features --deriv-path /chemin/vers/derivatives/preprocessed-noica \
               --save-path  /chemin/vers/dream_features \
               --n-jobs     $SLURM_CPUS_PER_TASK
```

Écrit des `.npz` par sujet, feature et stade de sommeil.

### 4. Classification

```bash
dream-classify --save-path /chemin/vers/dream_features \
               --n-jobs    $SLURM_CPUS_PER_TASK \
               --n-perm    1000 \
               --key       cov \
               --state     S2 \
               --checkpoint-every 50
```

Sans `--key` ni `--state`, toutes les features ou tous les stades sont
traités.

Chaque commande accepte `--help`. Les modules peuvent aussi être lancés
directement : `python -m dream_recall_alex.classify`.

### 5. Reproduction des perms d'arthur (optionnel pour mon pipeline)

Ces scripts consomment les sorties de l'étape 4. Ils ne font pas partie de la
chaîne de traitement : les résultats existent sans eux.

**Schéma de permutation d'Arthur.**

```bash
python scripts/replicate_arthur_ffx.py \
    --save-path /chemin/vers/dream_features \
    --n-jobs    $SLURM_CPUS_PER_TASK \
    --n-perm    1000 \
    --key       cov \
    --state     S2
```

Recalcule la distribution nulle avec le schéma de permutation epoch, en
réutilisant les bootstraps de l'étape 4. Écrit
`results/{key}_{state}_epochperm.npz` sans toucher au résultat existant. Voir
« Statistiques » pour la justification.

## Choix méthodologiques

Cette section documente les décisions qui ne se lisent pas dans le code. Le
détail de chacune est dans le docstring d'en-tête du fichier concerné.

### Données

38 sujets, EEG de sommeil nocturne (dataset Ruby/Eichenlaub, CRNL Lyon),
BrainAmp, 19 canaux EEG, 1000 Hz, référence nez.

- **High recallers** : sujets 1 à 18. **Low recallers** : 19, 20, 23 à 38.
- **Sujets 21 et 22 exclus** de l'analyse HR/LR (`EXCLUDED_SUBJECTS` dans
  `config.py`), mais prétraités quand même : les données peuvent servir à
  d'autres analyses.
- Stades notés selon Rechtschaffen & Kales (S1 à S4, REM), pas AASM. Les états
  de classification sont S2, SWS (S3+S4), NREM (S2+S3+S4) et REM
  (`CLASSIFICATION_GROUPS`).

### Prétraitement

Trois branches sont produites en parallèle, chacune dans son propre derivative
BIDS : `noica`, `ica`, `iclabel`.

- **`noica`** est la référence directe à Arthur, qui n'applique pas d'ICA.
- **`ica`** et **`iclabel`** sont des tentatives d'ameliorer la data avant le la donner au classifier.


**Pas de décimation.** `DECIMATE=False`, tout reste à 1000 Hz, ce qui reproduit
Arthur exactement. Un pipeline antérieur à 250 Hz s'écartait de la référence et
a été abandonné car aucune utilité réelle. Mais la decimation est conservé pour servir au pipeline de CNN d'anirudh.

**Pas de re-référencement.** La référence d'enregistrement (nez) est conservée.
La CAR a été utilisée puis retirée : elle introduit un projecteur qui réduit le
rang des matrices de covariance à 18 sur 19, ce qui les rend semi-définies
positives au lieu de définies positives strictes. Le logarithme matriciel de la
projection riemannienne diverge alors (`ValueError: Matrices must be positive
definite`, confirmé sur nos données). La référence nez garantit le rang plein
sans shrinkage.

**Filtrage.** Notch 50/100 Hz, puis passe-haut à 0.1 Hz, qui correspond au
filtre matériel déclaré dans le sidecar BIDS. L'ICA est ajustée sur une copie
filtrée à 1 Hz (elle converge mal à 0.1 Hz => reco mne ) puis appliquée aux données à
0.1 Hz.

### Features

Segmentation en epochs de 30 s par stade atomique (S1, S2, S3, S4, REM). Les
états composites sont obtenus par concaténation des `.npz` atomiques, sans
relecture ni recalcul.

**Overlap des fenêtres de Welch.** 50 % pour la PSD (`OVERLAP=500` sur des
fenêtres de 1000), 75 % pour le cospectre. La variance de l'estimateur de Welch
décroît en 1/K où K est le nombre de fenêtres moyennées ; à durée d'epoch fixe,
50 % de recouvrement double environ K. Le gain mesuré est réel mais faible
(par exemple `cosp_sigma/S2` : 69.18 % avec overlap contre 68.21 % sans et petite 
ameliroation de la pvalue subject) : l'overlap stabilise l'estimation,
il ne fabrique pas de signal. Pas de leakage.

**FOOOF en mode `fixed`, pas `knee`.** Validé empiriquement sur environ 139 000
spectres : ΔR² = −0.0004 et Δnpeaks = +0.17, très en deçà des seuils retenus
(0.02 et 0.8). Le mode `knee` n'apporte rien sur ces données.

**`psd_osc_*` en ratio linéaire** (`flat_psds / 10**ap_fit_log`), cohérent avec
la notion de power spectrum ratio de FOOOF (Donoghue et al. 2020). La
soustraction en espace log a été rejetée : elle produit des valeurs dans un
espace incomparable aux features `psd_*`.

**Covariances par SCM**, (sans shrinkage OAS ou Ledoit-Wolf), cohérent avec
Arthur. Une régularisation numérique (diagonal loading 1e-10) garantit la
positivité stricte.

**Complexité via antropy** (permutation entropy, spectral entropy, Higuchi
fractal dimension). LZC volontairement non implémentée, trop corrélée à la
pente spectrale donc repetitif. Ces mesures sont à comparer systématiquement à l'exposant
aperiodic seul : une mesure de complexité peut n'être qu'une remesure de la
pente 1/f (Aamodt et al. 2022).

### Classification

**Deux modes selon la géométrie de la feature.** Les matrices (`cov`, `cosp_*`)
sont classifiées par `TSclassifier(LDA())` en espace tangent riemannien. Les
vecteurs (`psd_*`, `psd_osc_*`, `aperiodic`, complexité) par un LDA euclidien,
une électrode à la fois. La liste des features matricielles est déclarée dans
`config.MATRIX_KEYS`.

**Validation croisée** : `StratifiedLeave2GroupsOut`, un sujet HR et un sujet
LR en test à chaque split, soit 324 splits (18 × 18).

**Équilibrage** : `n_trials_min` epochs tirés par sujet, sans remise, identique
pour toutes les features et tous les stades (`compute_global_n_trials`, calculé
sur la feature de référence `cov`). 1000 bootstraps.

**Pas de standardisation.** La classification vectorielle est univariée : un
LDA par électrode, sur une seule colonne. `StandardScaler` y serait une
transformation affine sur l'unique feature, et l'objectif de LDA est invariant
aux transformations linéaires non-singulières. Vérifié : run complet avec
standardisation, delta d'accuracy = 0.0000 exactement sur les 51 combos
comparables. Sur les matrices la question ne se pose pas de la même façon, le
whitening riemannien joue déjà ce rôle. Arthur ne standardise pas non plus.

### Statistiques

**Permutations au niveau sujet.** C'est la principale divergence avec le code
de référence. Arthur permute les labels au niveau des epochs (`utils.py:103`) :
le même index de permutation est appliqué à `y` et à `groups`, si bien que
chaque sujet permuté devient un paquet aléatoire d'epochs des deux classes.
L'hypothèse nulle testée porte alors sur l'échantillon d'epochs, pas sur la
population de sujets, et la distribution nulle s'en trouve resserrée. Nous
permutons au niveau des sujets, ce qui teste la généralisation à la population
(Combrisson et al. 2022, NeuroImage).

Les deux schémas sont implémentés et rapportés côte à côte plutôt que
d'affirmer l'écart sans le mesurer. Le schéma sujet est celui du pipeline
principal (`classify.py`, `permute_subject_labels`) ; le schéma epoch est
obtenu par `scripts/replicate_arthur_ffx.py`, qui réutilise les bootstraps déjà
calculés, ceux-ci ne dépendent pas du schéma de permutation, et ne recalcule
que la distribution nulle. Les sorties coexistent dans `results/` :

| Fichier | Schéma |
|---|---|
| `{feature}_{stade}.npz` | sujet (pipeline principal) |
| `{feature}_{stade}_epochperm.npz` | epoch (réplication Arthur) |

Écart mesuré : le schéma epoch déflate les p-values d'un facteur ~80 en médiane
par rapport au schéma sujet.

**Trois niveaux de correction** pour les comparaisons multiples, rapportés dans
`pvalue_summary_table.csv` :

| Niveau | Portée |
|---|---|
| non corrigé | un test |
| max-stat par bande | 19 électrodes, correspond à ce que fait Arthur |
| max-stat pooled | famille complète de features (notre extension) |

Le mode par bande n'a pas de sens pour les features matricielles, qui ne
produisent qu'une mesure : ces lignes affichent `N/A` plutôt qu'un chiffre
inventé.

**Reproductibilité.** Les graines dérivent d'un hachage MD5 (`_seed`), et non
de `hash()`, dont le résultat dépend de `PYTHONHASHSEED` et change entre deux
exécutions.

### Résultats négatifs documentés

**Riemannian Potato.** Filtre d'artefacts testé sur quatre configurations, avec
un design à trois bras pour séparer l'effet du filtre de celui de la perte
d'epochs qu'il entraîne. Les p-values sont identiques à la troisième décimale
dans toutes les conditions : aucun gain mesurable. Écarté.

**Exhaustive Feature Selection.** Combiner les features survivantes n'améliore
pas la meilleure feature seule. `cosp_sigma` domine en S2, `cosp_delta` en SWS.
Le signal est concentré dans un cospectre par stade, pas distribué entre
plusieurs mesures.

## Structure du dépôt

```
src/dream_recall_alex/           le package : le pipeline
    config.py                    canaux, labels sujets, stades, bandes
    utils.py                     primitives bas niveau partagées
    mat_eeg_to_bids.py           étape 1
    preprocess_subject.py        étape 2
    feat_extract_umap_fooof.py   étape 3
    classify.py                  étape 4
scripts/                         analyses et figures, consomment le package
    replicate_arthur_ffx.py      schéma de permutation d'Arthur (contrôle FFX)
archive/                         pipeline original d'Arthur, conservé pour référence
```
