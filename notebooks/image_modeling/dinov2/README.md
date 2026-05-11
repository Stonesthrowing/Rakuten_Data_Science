# DINOv2 — Frozen Backbone Classifier

> ⚠️ **Warning: training stopped before convergence.**
> Each epoch took ~60 minutes on Apple Silicon MPS. Due to the high compute cost,
> training was stopped early and the model had not yet converged. Reported metrics
> (macro F1 0.6199) should be treated as a lower bound, not a final result.

Pretrained `vit_small_patch14_dinov2` loaded via `timm`. The backbone is **fully frozen**;
only a linear classification head is trained on top of the extracted features.

## Notebook

| Notebook | Augmentation | Best Macro F1 | Best Accuracy |
|----------|-------------|---------------|---------------|
| `model_i13_dinov2_train_aug_frozen.ipynb` | timm training transform | 0.6199 | 0.6647 |

## Run order

```
1. model_i13_dinov2_train_aug_frozen.ipynb   ← single notebook, self-contained
```

## Freeze strategy

```
backbone (vit_small_patch14_dinov2)  → fully frozen (requires_grad = False)
classification head                  → trainable linear layer
```

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| `DINO_MODEL_NAME` | `vit_small_patch14_dinov2` |
| `IMAGE_SIZE` | 224 |
| `BATCH_SIZE` | 8 |
| `INITIAL_LR` | 1e-4 |
| `WEIGHT_DECAY` | 1e-4 |
| `MAX_EPOCHS` | 3 |
| `EARLY_STOPPING_PATIENCE` | 2 |
| `SEED` | 42 |
| Scheduler | ReduceLROnPlateau |
| Augmentation | timm `resolve_model_data_config` default training transform |

## Outputs

Saves to `outputs/image_modeling/model_i13_dinov2_train_aug_frozen/`:

- `history.csv` — per-epoch train/val metrics
- `val_predictions.csv` — per-sample predictions at best epoch
- `run_metadata.json` — hyperparameters and final metrics
