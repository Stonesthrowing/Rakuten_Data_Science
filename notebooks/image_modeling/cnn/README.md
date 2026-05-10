# CNN Baseline Models — Trained from Scratch

Custom convolutional networks built and trained from scratch (no pretrained weights).
Architecture: 4 conv blocks (3→32→64→128→256 channels, 3×3 kernels), each followed by
BatchNorm, ReLU, and MaxPool — then a fully-connected head.

> These models serve as the **lower-bound baseline** before transfer learning experiments.
> Best macro F1 reaches ~0.509, well below pretrained models (ResNet ≥ 0.653).

## Notebooks

| Notebook | Image size | Augmentation | Best Macro F1 | Best epoch |
|----------|-----------|--------------|---------------|------------|
| `model_i1_cnn128_noaug_from_scratch_lower_LR.ipynb` | 128 px | No | 0.5065 | 30 |
| `model_i2_cnn128_moderateAug_from_scratch.ipynb` | 128 px | Moderate | 0.4984 | 45 |
| `model_i3_cnn256_noAug_from_scratch.ipynb` | 256 px | No | 0.5090 | 33 |

## Run order

```
1. model_i1_cnn128_noaug_from_scratch_lower_LR.ipynb   ← 128 px baseline, no augmentation
2. model_i2_cnn128_moderateAug_from_scratch.ipynb       ← same resolution, adds augmentation
3. model_i3_cnn256_noAug_from_scratch.ipynb             ← higher resolution probe (256 px)
```

Each notebook is self-contained. All hyperparameters are defined in the first cell.

## Key hyperparameters

| Parameter | I1 (128, no aug) | I2 (128, aug) | I3 (256, no aug) |
|-----------|-----------------|--------------|-----------------|
| `IMAGE_SIZE` | 128 | 128 | 256 |
| `BATCH_SIZE` | 128 | 128 | 64 |
| `INITIAL_LR` | 5e-4 | 5e-4 | — |
| `WEIGHT_DECAY` | 1e-4 | 1e-4 | — |
| `MAX_EPOCHS` | 45 | 45 | 35 |
| `EARLY_STOPPING_PATIENCE` | 15 | 15 | 10 |
| `SEED` | 42 | 42 | 42 |

## Augmentation (I2 only)

```
RandomResizedCrop(128, scale=(0.85, 1.0))
RandomHorizontalFlip
ColorJitter (mild)
```

## Outputs

Each notebook saves to `outputs/image_modeling/<LOCAL_RUN_NAME>/`:

- `history.csv` — per-epoch train/val metrics
- `val_predictions.csv` — per-sample predictions at best epoch
- `confusion_matrix.png` — confusion matrix figure
- `run_metadata.json` — hyperparameters and final metrics
