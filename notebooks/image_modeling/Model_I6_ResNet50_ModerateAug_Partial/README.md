# Model I6 — ResNet50, Moderate Augmentation, Partial Fine-Tuning

ResNet50 classifier with a **frozen backbone** — only the classification head is trained.
Faster to train than full fine-tuning but less expressive.

## Run order

```
1. config.py     ← review / adjust hyperparameters before anything else
2. train.py      ← trains the model, saves best checkpoint
3. evaluate.py   ← loads best checkpoint, prints metrics, saves confusion matrix
```

`dataset.py` and `utils.py` are imported automatically by the scripts above — do not run them directly.

## Where to adjust the training setup

All hyperparameters live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LEARNING_RATE` | `3e-4` | Head learning rate |
| `BATCH_SIZE` | `128` | Training batch size |
| `IMAGE_SIZE` | `224` | Input resolution |
| `MAX_EPOCHS` | `18` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `6` | Epochs without improvement before stopping |
| `WEIGHT_DECAY` | `1e-4` | L2 regularisation |
| `SEED` | `42` | Reproducibility seed |

Data paths and the output directory (`RUN_NAME = "I6_ResNet50_ModerateAug_Partial"`) are also set in `config.py`.
