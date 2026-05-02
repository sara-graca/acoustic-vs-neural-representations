# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

This repository contains the full analysis pipeline for a study comparing acoustic and neural speech representations in a French L1/L2 corpus.

## Overview

The project investigates whether acoustic features (F1, F2, F3, f0) and neural representations (Whisper, XLS-R) encode the phonological structure of French in similar ways, with a focus on L1/L2 differences between native French and native Russian speakers learning French.

## Pipeline

The pipeline is managed with DVC. All stages are defined in `dvc.yaml` and all tunable parameters in `params.yaml`.

| Stage | Script | Description |
|-------|--------|-------------|
| Parsing | `parse_corpus.py` | Extracts phoneme tokens from TextGrids |
| Acoustic extraction | `extract_acoustics.py` | Extracts F1, F2, F3, f0, SCG via Parselmouth |
| Neural extraction | Kaggle notebooks | Whisper and XLS-R embeddings (GPU required) |
| Normalisation | `normalise.py` | Lobanov normalisation + PCA on neural features |
| Descriptive acoustic | `descriptive_acoustic.py` | Vowel charts, boxplots, violin plots |
| Descriptive neural | `descriptive_neural.py` | PCA/UMAP projections, variance ratios, cosine similarity |
| Cross-representation | `descriptive_crossrep.py` | RSMs and Mantel tests across representations |
| Acoustic L1/L2 tests | `tests_acoustic_l1l2.py` | Mann-Whitney/t-tests with FDR correction |
| Acoustic gender tests | `tests_acoustic_gender.py` | Residual gender effect after Lobanov |
| Neural L1/L2 tests | `tests_neural_l1l2.py` | Permutation tests on cosine distances |
| Distance analysis | `tests_inter_distances.py` | Distance matrices, Mantel tests, bootstrap CIs |
| Classifier | `nearest_centroid_classifier.py` | LOSO nearest-centroid classifier + McNemar tests |
| LME models | `lme_models.py` | Linear mixed-effects models |
| ROPE analysis | `ci_rope.py` | Profile likelihood CIs and ROPE classification |
| Vowel clustering | `clustering_vowels.py` | Hierarchical clustering of vowels and consonants |
| Speaker clustering | `clustering_speakers.py` | Hierarchical clustering of speakers |

## Setup

```bash
# Install dependencies
pixi install

# Reproduce the full pipeline
dvc repro
```

## Notes

- Neural feature extraction requires GPU and was run on Kaggle. Outputs are tracked manually with DVC.
- All parameters (ROPE bounds, bootstrap iterations, PCA dimensions, etc.) are in `params.yaml`.
- Results are saved to `results/tables/` and `results/figures/`.

## Requirements

- Python 3.10+
- pixi
- DVC
- Parselmouth
- GPU access for neural extraction (Whisper medium, XLS-R 300M)
