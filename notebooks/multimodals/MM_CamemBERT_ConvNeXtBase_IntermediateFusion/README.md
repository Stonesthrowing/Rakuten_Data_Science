# MM — CamemBERT + ConvNeXt-Base, Intermediate Fusion

Trains a lightweight **MLP fusion head** on top of pre-extracted image and text features.
No image/text model is retrained here — the heavy lifting is done by I12 and T8.

## Prerequisites

The following exported feature files must exist before running anything here:

| Source model | Required files |
|---|---|
| **I12** (ConvNeXt-Base) | `train_features_1024d.npy`, `val_features_1024d.npy` |
| **T8** (CamemBERT) | `text_train_features_768d.npy`, `text_val_features_768d.npy` |

Run `I12/train.py` and `T8/train.py` (or their `recover_*.py` scripts) first.

## Run order

```
1. config.py          ← review / adjust hyperparameters and feature paths
2. train.py           ← trains MLP fusion head on concatenated 1792-d features (1024 + 768)
3. evaluate.py        ← loads best checkpoint, prints metrics, saves confusion matrix
4. analyze_errors.py  ← (optional) finds samples where the model is most confidently wrong
5. gradcam.py         ← (optional) Grad-CAM visualisation for a specific sample
```

`model.py`, `dataset.py`, and `utils.py` are imported automatically — do not run them directly.

### Optional analysis scripts

**`analyze_errors.py`** — prints and saves a CSV of the top-N validation samples where the
fusion model is most confidently incorrect. Use the output to pick a `GRADCAM_TARGET_INDEX`
for `gradcam.py`.

**`gradcam.py`** — runs Grad-CAM on a single validation sample to show which image regions
drove the prediction. Requires the `grad-cam` and `timm` packages.
Workflow: run `analyze_errors.py` → set `GRADCAM_TARGET_INDEX` in `config.py` → run `gradcam.py`.

## Where to adjust the training setup

All hyperparameters and paths live in **`config.py`**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `INPUT_DIM` | `1792` | Concatenated feature size (1024 image + 768 text) — change if swapping backbones |
| `HIDDEN_DIM_1` | `256` | First MLP hidden layer width |
| `HIDDEN_DIM_2` | `128` | Second MLP hidden layer width |
| `DROPOUT_1` | `0.5` | Dropout after first hidden layer |
| `DROPOUT_2` | `0.4` | Dropout after second hidden layer |
| `LEARNING_RATE` | (see config) | Fusion head learning rate |
| `BATCH_SIZE` | (see config) | Training batch size |
| `MAX_EPOCHS` | (see config) | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | (see config) | Epochs without improvement before stopping |
| `GRADCAM_TARGET_INDEX` | (see config) | Validation sample index for Grad-CAM |

Feature file paths and the output directory
(`RUN_NAME = "MM_CamemBERT_ConvNeXtBase_IntermediateFusion"`) are also in `config.py`.
