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

## Structure du dépôt

```
src/dream_recall_alex/
    config.py                    canaux, labels sujets, stades, bandes
    utils.py                     primitives bas niveau partagées
    mat_eeg_to_bids.py           étape 1
    preprocess_subject.py        étape 2
    feat_extract_umap_fooof.py   étape 3
    classify.py                  étape 4
archive/                         pipeline original d'Arthur, conservé pour référence
```
