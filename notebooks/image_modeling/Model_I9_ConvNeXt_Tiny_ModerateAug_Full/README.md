# Model I9 — ConvNeXt-Tiny, Moderate Augmentation, Full Fine-Tuning

ConvNeXt-Tiny classifier with **full fine-tuning**, **Automatic Mixed Precision (AMP)**, and
**differential learning rates** (backbone gets a lower LR than the head).
Uses TrivialAugmentWide for data augmentation.

## Run order

```
1. config.py     ← review / adjust hyperparameters before anything else
2. train.py      ← trains the model with AMP, saves best checkpoint
3. evaluate.py   ← loads best checkpoint, prints metrics, saves confusion matrix
```

`dataset.py` and `utils.py` are imported automatically — do not run them directly.

## Where to adjust the training setup

All hyperparameters live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BACKBONE_LR` | `1e-5` | Learning rate for ConvNeXt backbone layers |
| `HEAD_LR` | `1e-4` | Learning rate for the classification head |
| `BATCH_SIZE` | `32` | Smaller than ResNet50 due to model memory footprint |
| `IMAGE_SIZE` | `224` | Input resolution |
| `MAX_EPOCHS` | `18` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `6` | Epochs without improvement before stopping |
| `WEIGHT_DECAY` | `0.05` | Higher than ResNet50 (ConvNeXt default) |
| `SEED` | `42` | Reproducibility seed |

Data paths and the output directory (`RUN_NAME = "I9_ConvNeXt_Tiny_ModerateAug_Full"`) are also set in `config.py`.
