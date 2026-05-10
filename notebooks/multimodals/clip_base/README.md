# CLIP Base — Multimodal Classifier (CLIP Text + CLIP Vision)

Both text and image branches come from `openai/clip-vit-base-patch32`. CLIP's own text encoder
produces a 512-d embedding; CLIP Vision produces a 512-d pooled embedding. The two are
L2-normalised, concatenated (1024-d), and passed through a 2-layer MLP head.

> **Note:** CLIP's text encoder is limited to 77 tokens and was trained on English image-text
> pairs. On this French-language dataset it underperforms CamemBERT significantly.

## Notebooks

| Notebook | Backbone freeze | Best macro F1 |
|----------|----------------|---------------|
| `mm_clip_base_noaug_frozen.ipynb` | Fully frozen | 0.6723 |
| `mm_clip_base_noaug_partial_unfreeze.ipynb` | Partial unfreeze | — |

## Run order

```
1. mm_clip_base_noaug_frozen.ipynb        ← start here; frozen backbone, fastest to run
2. mm_clip_base_noaug_partial_unfreeze.ipynb  ← optional; unfreezes last layers of CLIP
```

Each notebook is self-contained — no external config file. All hyperparameters are defined
in the first cell.

## Key hyperparameters (frozen notebook)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `FREEZE_CLIP` | `True` | Freezes all CLIP parameters |
| `BATCH_SIZE` | `8` | Effective batch = 8 × ACCUM_STEPS |
| `ACCUM_STEPS` | `2` | Gradient accumulation steps |
| `INITIAL_LR` | `1e-5` | Learning rate for the MLP head only |
| `MAX_EPOCHS` | `15` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `3` | Epochs without improvement before stopping |
| `MAX_TEXT_LEN` | `77` | CLIP hard token limit |

## Outputs

Each notebook saves to `outputs/image_modeling/<LOCAL_RUN_NAME>/`:

- `history.csv` — per-epoch train/val metrics
- `val_predictions.csv` — per-sample predictions at best epoch
- `y_logits.npy` / `y_proba.npy` — logits and softmax probabilities (for late fusion)
- `run_metadata.json` — hyperparameters and final metrics
