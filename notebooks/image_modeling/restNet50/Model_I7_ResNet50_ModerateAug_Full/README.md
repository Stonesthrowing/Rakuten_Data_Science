# Model I7 — ResNet50, Moderate Augmentation, Full Fine-Tuning (restNet50 variant)

Same model as `notebooks/image_modeling/Model_I7_ResNet50_ModerateAug_Full` but located
in the `restNet50/` subdirectory and using absolute imports.
ResNet50 with **all layers trainable** and a lower learning rate to protect pretrained weights.

## Run order

```
1. config.py     ← review / adjust hyperparameters before anything else
2. train.py      ← trains the model, saves best checkpoint
3. evaluate.py   ← loads best checkpoint, prints metrics, saves confusion matrix
```

`dataset.py` and `utils.py` are imported automatically — do not run them directly.

## Where to adjust the training setup

All hyperparameters live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LEARNING_RATE` | `1e-4` | Lower than I6 to avoid destroying pretrained weights |
| `BATCH_SIZE` | `128` | Training batch size |
| `IMAGE_SIZE` | `224` | Input resolution |
| `MAX_EPOCHS` | `18` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `6` | Epochs without improvement before stopping |
| `WEIGHT_DECAY` | `1e-4` | L2 regularisation |
| `SEED` | `42` | Reproducibility seed |

Data paths and the output directory are set in `config.py`.
