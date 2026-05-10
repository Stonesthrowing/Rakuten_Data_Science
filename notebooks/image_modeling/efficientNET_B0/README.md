# EfficientNet-B0 — Partial Fine-Tuning

Pretrained `efficientnet_b0` (ImageNet weights) with **partial fine-tuning**: all backbone
layers are frozen except the last feature block (`features[-1]`) and the classification head,
which is replaced with a single linear layer (`in_features → 27 classes`).

## Notebooks

| Notebook | Augmentation | Best Macro F1 | Best Accuracy |
|----------|-------------|---------------|---------------|
| `model_i10_efficientNetB0_noaug_partial.ipynb` | No | 0.5684 | 0.6173 |
| `model_i11_efficientNetB0_moderateaug_partial.ipynb` | Moderate | 0.5489 | 0.5990 |

## Run order

```
1. model_i10_efficientNetB0_noaug_partial.ipynb       ← no augmentation baseline
2. model_i11_efficientNetB0_moderateaug_partial.ipynb  ← adds moderate augmentation
```

Each notebook is self-contained. All hyperparameters are defined in the first cell.

## Freeze strategy

```
backbone (features[0..6])  → frozen
features[-1]               → trainable
classifier head            → replaced and trainable
```

Only the last feature block and the new head receive gradient updates.

## Key hyperparameters

| Parameter | I10 (no aug) | I11 (aug) |
|-----------|-------------|-----------|
| `IMAGE_SIZE` | 224 | 224 |
| `BATCH_SIZE` | 64 | 32 |
| `INITIAL_LR` | 3e-4 | 3e-4 |
| `MIN_LR` | 1e-6 | 1e-6 |
| `MAX_EPOCHS` | 18 | 18 |
| `EARLY_STOPPING_PATIENCE` | 6 | 6 |
| `SEED` | 42 | 42 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

## Augmentation (I11 only)

```
RandomResizedCrop(224, scale=(0.85, 1.0))
RandomHorizontalFlip
ColorJitter (mild)
```

## Outputs

Each notebook saves to `outputs/image_modeling/<LOCAL_RUN_NAME>/`:

- `history.csv` — per-epoch train/val metrics
- `val_predictions.csv` — per-sample predictions at best epoch
- `confusion_matrix.png` — confusion matrix figure
- `run_metadata.json` — hyperparameters and final metrics
