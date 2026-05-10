# CamemBERT + CLIP Vision — Gated Fusion

Replaces CLIP's English text encoder with **CamemBERT** (French pretrained) while keeping
**CLIP Vision** (`openai/clip-vit-base-patch32`) as the image backbone. A sigmoid gating
network learns per-feature mixing weights between the two 768-d embeddings end-to-end.

## Folder structure

```
frozen/                          ← both backbones fully frozen during training
    mm_camembert_clip_gated_fusion_frozen.ipynb

unfrozen/                        ← 3-stage training (head → partial → full unfreeze)
    mm_camembert_clip_gated_fusion_staged_unfreeze.ipynb

fine_tuned_with_aug/             ← staged unfreeze + image augmentation + label smoothing
    mm_camembert_clip_aug_gatedfusion_unfreeze.ipynb

unfrozen_better_algo/            ← projection layers + softmax gate + aug + label smoothing
    CLIP_camemBERT_partially_unfrozen_with_aug.ipynb
```

## Run order (recommended)

```
1. frozen/mm_camembert_clip_gated_fusion_frozen.ipynb
       ← fastest baseline; no backbone training required

2. unfrozen/mm_camembert_clip_gated_fusion_staged_unfreeze.ipynb
       ← adds staged unfreezing (stage1: head only → stage2: partial → stage3: full)

3. fine_tuned_with_aug/mm_camembert_clip_aug_gatedfusion_unfreeze.ipynb
       ← same staged unfreeze + image augmentation + label smoothing

4. unfrozen_better_algo/CLIP_camemBERT_partially_unfrozen_with_aug.ipynb
       ← adds linear projection layers (768→512) and softmax gate; best architecture
```

## Results summary

| Notebook | Augmentation | Label smoothing | Best macro F1 |
|----------|-------------|-----------------|---------------|
| frozen | No | No | 0.8640 |
| unfrozen (staged) | No | No | 0.8644 |
| fine_tuned_with_aug | Yes | No | 0.8802 |
| unfrozen_better_algo | Yes | 0.1 | — |

## Key hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `CAMEMBERT_ID` | `camembert-base` | French pretrained text encoder |
| `CLIP_ID` | `openai/clip-vit-base-patch32` | Image encoder |
| `MAX_TEXT_LEN` | `128` | CamemBERT supports longer sequences than CLIP |
| `BATCH_SIZE` | `16` | Effective batch = 16 × ACCUM_STEPS |
| `ACCUM_STEPS` | `2` | Gradient accumulation steps |
| `EARLY_STOPPING_PATIENCE` | `5` | Epochs without improvement before stopping |
| `LABEL_SMOOTHING` | `0.1` | Only in `unfrozen_better_algo` notebook |
| `PROJECTION_DIM` | `512` | Only in `unfrozen_better_algo` notebook |

## Staged unfreezing schedule

| Stage | Epochs | Backbone LR | Head LR |
|-------|--------|-------------|---------|
| stage1 — head only | 3 | 0.0 | 5e-4 |
| stage2 — partial unfreeze (layers 10–11) | 4 | 2e-6 | 1e-4 |
| stage3 — full unfreeze | 8 | 1e-6 | 5e-5 |

## Outputs

Each notebook saves to `outputs/<RUN_NAME>/`:

- `training_history.csv` — per-epoch metrics including gate weights
- `best_val_logits.npy` — logits at best epoch (for late fusion / stacking)
- `best_val_gate_weights.npy` — per-sample text/image gate weights
- `best_metadata.json` — hyperparameters and final metrics
