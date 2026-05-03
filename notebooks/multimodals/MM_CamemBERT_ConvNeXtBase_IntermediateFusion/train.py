# ============================================================
# train.py
# Training script for MM_CamemBERT_ConvNeXtBase_IntermediateFusion.
#
# Trains a lightweight MLP fusion head on pre-extracted features.
# All hyperparameters and paths are defined in config.py.
#
# Prerequisites:
#   - I12 feature export: train/val_features_1024d.npy
#   - T8  feature export: text_train/val_features_768d.npy
# ============================================================

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import RakutenFusionDataset
from model import IntermediateFusionHead
from utils import (
    plot_and_save_history,
    run_epoch,
    save_history_json,
    save_run_metadata,
)


def main() -> None:

    # ----------------------------------------------------------
    # 1. Setup
    # ----------------------------------------------------------
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run    : {config.RUN_NAME}")
    print(f"Device : {device}")
    print(f"Input  : {config.INPUT_DIM}d  ({config.IMG_FEATURE_DIM}d image + {config.TXT_FEATURE_DIM}d text)")

    for d in [config.LOCAL_OUTPUT_ROOT, config.LOCAL_MODEL_ROOT, config.LOCAL_FIG_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 2. Load splits & label mapping
    # ----------------------------------------------------------
    train_df = pd.read_csv(config.SPLIT_DIR / "train_split.csv")
    val_df   = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    print(f"Train  : {len(train_df):,}  |  Val: {len(val_df):,}")

    # ----------------------------------------------------------
    # 3. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_ds = RakutenFusionDataset(
        config.TRAIN_FEATURES_IMAGE, config.TRAIN_FEATURES_TEXT, train_df, label2id
    )
    val_ds = RakutenFusionDataset(
        config.VAL_FEATURES_IMAGE, config.VAL_FEATURES_TEXT, val_df, label2id
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # ----------------------------------------------------------
    # 4. Model, loss, optimizer
    # ----------------------------------------------------------
    model     = IntermediateFusionHead().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {trainable:,} trainable")

    # ----------------------------------------------------------
    # 5. Training loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Training  (Intermediate Fusion MLP) …")
    print("=" * 60)
    start_time = time.time()

    history: list          = []
    best_macro_f1: float   = -float("inf")
    best_epoch: int        = -1
    epochs_no_improve: int = 0

    for epoch in range(1, config.MAX_EPOCHS + 1):

        train_loss, train_acc, train_macro_f1, train_weighted_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        epoch_result = {
            "epoch":             epoch,
            "train_loss":        float(train_loss),
            "val_loss":          float(val_loss),
            "train_acc":         float(train_acc),
            "val_acc":           float(val_acc),
            "train_macro_f1":    float(train_macro_f1),
            "val_macro_f1":      float(val_macro_f1),
            "train_weighted_f1": float(train_weighted_f1),
            "val_weighted_f1":   float(val_weighted_f1),
        }
        history.append(epoch_result)
        save_history_json(history, config.HISTORY_JSON_LOCAL)

        print(
            f"Epoch {epoch:>2}/{config.MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}  val_macro_f1={val_macro_f1:.4f}  "
            f"val_weighted_f1={val_weighted_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1  = val_macro_f1
            best_epoch     = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.BEST_WEIGHTS_LOCAL)
            print("    >>> New best model saved!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print("    >>> Early stopping triggered.")
            break

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    total_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("Training Finished")
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Best F1     : {best_macro_f1:.4f}")
    print(f"  Duration    : {total_minutes:.1f} min")
    print("=" * 60)

    save_run_metadata(
        config.METADATA_JSON_LOCAL,
        model_name=config.RUN_NAME,
        best_f1=best_macro_f1,
        best_epoch=best_epoch,
        duration_min=total_minutes,
    )
    plot_and_save_history(history, config.LOCAL_FIG_ROOT, best_epoch, best_macro_f1)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
