# Experiments

This folder contains all Jupyter notebooks used throughout the project, 
including final model runs, hyperparameter sweeps, ablation studies, and 
evaluation scripts.

## Main Experiments

| Notebook | Description |
|----------|-------------|
| `FINAL_CTGAN.ipynb` | Final CTGAN experiment. Trains CTGAN under natural and forced fraud ratios, evaluates hybrid datasets using XGBoost, and includes t-SNE visualisation, KS/Wasserstein distance, and privacy evaluation via NN memorisation. |
| `FINAL_DIFF.ipynb` | Final FinDiff experiment. Mirrors the structure of FINAL_CTGAN using a diffusion-based generative model. Ratio forcing is implemented by directly manipulating the label array in `sample_diffusion()`. |

## Hyperparameter Experiments

| Notebook | Description |
|----------|-------------|
| `CTparam_1_.ipynb` | CTGAN hyperparameter sweep conducted on Google Colab. Varies epochs, batch size, embedding dimension, log frequency, and discriminator steps. |
| `CTGAN_batch_dstep.ipynb` | Local re-run of CTGAN sweep focusing on batch size and discriminator steps. Complements CTparam_1_ with additional configurations. |

## Ablation Studies

| Notebook | Description |
|----------|-------------|
| `QuantileTran_Ver.ipynb` | FinDiff scaler ablation. Replaces the default StandardScaler with QuantileTransformer (n_quantiles=1000, output_distribution="normal") in both the generative pipeline and XGBoost evaluation. |

## Ratio & Performance Analysis

| Notebook | Description |
|----------|-------------|
| `GAN_ratio_per_.ipynb` | Early-stage CTGAN ratio performance analysis. Predecessor to FINAL_CTGAN, used to explore the effect of train/test data ratios. |
| `findiff_ratio_performance_.ipynb` | Early-stage FinDiff ratio performance analysis. Mirrors GAN_ratio_per_ for the diffusion model. |

## Evaluation & Visualisation

| Notebook | Description |
|----------|-------------|
| `ConfusionMat_GAN_.ipynb` | Generates confusion matrices for all CTGAN experimental cases including F1 score breakdown. |
| `ConfusionMat_diff_.ipynb` | Generates confusion matrices for all FinDiff cases. Also identifies the double penalisation issue when using sample_weight with natural ratio. |

## External Model Sources

| File | Description |
|------|-------------|
| `ctgan.zip` | Custom-modified CTGAN source code forked from the original library. Adds ratio parameter support for forced minority class generation. Sourced from the original SDV paper implementation. |
| `tabsyn_.zip` | TabSyn model used in supplementary comparison experiments on a different dataset configuration. |

## Data Split Structure

All experiments follow a consistent train/validation/test split:
```
Full dataset (284,807)
├── test_df        → 20% (held out, never seen during training)
└── train_df       → 80%
    ├── val_df     → 20% of train_df (used for evaluation)
    └── train_base → 80% of train_df (~182,276 rows, used for generation)
```

Synthetic data is generated from `train_base` only. The classifier is 
then evaluated on `val_df` and `test_df` which contain only real data.