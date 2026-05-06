# ============================================================
# train.py
# Generic training script for Rakuten image classification.
# All hyperparameters and paths are defined in config.py.
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
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import notebooks.image_modeling.restNet50.Model_I6_ResNet50_ModerateAug_Partial.config as config
from notebooks.image_modeling.restNet50.Model_I6_ResNet50_ModerateAug_Partial.dataset import RakutenImageDataset
from notebooks.image_modeling.restNet50.Model_I6_ResNet50_ModerateAug_Partial.utils import (
    load_full_checkpoint,
    plot_and_save_history,
    run_epoch,
    save_full_checkpoint,
    save_history_json,
)


def main() -> None:

    # ----------------------------------------------------------
    # 1. Setup: memory, reproducibility, device
    # ----------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run     : {config.RUN_NAME}")
    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------
    # 2. Create output directories
    # ----------------------------------------------------------
    for d in [
        config.LOCAL_OUTPUT_ROOT,
        config.LOCAL_MODEL_ROOT,
        config.LOCAL_FIG_ROOT,
        config.SPLIT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 3. Load splits & label mapping
    # ----------------------------------------------------------
    train_df = pd.read_csv(config.SPLIT_DIR / "train_split.csv")
    val_df   = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    label2id_path = config.SPLIT_DIR / "label2id.json"
    if label2id_path.exists():
        with open(label2id_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        label2id = {int(k): int(v) for k, v in label2id.items()}
    else:
        labels   = sorted(train_df["prdtypecode"].unique())
        label2id = {int(label): i for i, label in enumerate(labels)}

    train_df["label_id"] = train_df["prdtypecode"].map(label2id)
    val_df["label_id"]   = val_df["prdtypecode"].map(label2id)

    def make_local_image_path(row, split: str = "train") -> str:
        fname = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        base  = config.LOCAL_IMAGE_TRAIN_DIR if split == "train" else config.LOCAL_IMAGE_TEST_DIR
        return str(base / fname)

    train_df["image_path_local"] = train_df.apply(make_local_image_path, axis=1)
    val_df["image_path_local"]   = val_df.apply(make_local_image_path, axis=1)

    num_classes = len(label2id)
    print(f"Classes : {num_classes}")
    print(f"Train   : {len(train_df):,}  |  Val: {len(val_df):,}")

    # ----------------------------------------------------------
    # 4. Transforms  (Moderate augmentation)
    # ----------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 20, config.IMAGE_SIZE + 20)),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ----------------------------------------------------------
    # 5. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_dataset = RakutenImageDataset(train_df, transform=train_transform)
    val_dataset   = RakutenImageDataset(val_df,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # ----------------------------------------------------------
    # 6. Model  (ResNet50, partially unfrozen: layer4 + head)
    # ----------------------------------------------------------
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze all layers …
    for param in model.parameters():
        param.requires_grad = False

    # … then unfreeze layer4 and the classification head
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(config.DROPOUT),
        nn.Linear(model.fc.in_features, num_classes),
    )
    # fc is unfrozen by default (newly created)

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params  : {trainable:,} trainable / {total:,} total")

    # ----------------------------------------------------------
    # 7. Loss, Optimizer, Scheduler
    # ----------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    # ----------------------------------------------------------
    # 8. Resume or initialise training state
    # ----------------------------------------------------------
    start_epoch             = 1
    history: list           = []
    best_macro_f1: float    = -float("inf")
    best_epoch: int         = -1
    epochs_no_improve: int  = 0

    if config.RESUME_TRAINING:
        ckpt_map = {
            "local_last": config.LAST_CKPT_LOCAL,
            "local_best": config.BEST_CKPT_LOCAL,
        }
        ckpt_path = ckpt_map[config.CHECKPOINT_SOURCE]
        if ckpt_path.exists():
            start_epoch, history, best_macro_f1, best_epoch = load_full_checkpoint(
                ckpt_path, model, optimizer, scheduler, device
            )
            print(f"Resumed from: {ckpt_path}  (starting epoch {start_epoch})")
        else:
            print(f"Checkpoint not found: {ckpt_path}  – starting from scratch.")

    # ----------------------------------------------------------
    # 9. Training loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Training …")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, config.MAX_EPOCHS + 1):

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        scheduler.step(val_f1)

        epoch_result = {
            "epoch":          epoch,
            "train_loss":     float(train_loss),
            "val_loss":       float(val_loss),
            "train_acc":      float(train_acc),
            "val_acc":        float(val_acc),
            "train_macro_f1": float(train_f1),
            "val_macro_f1":   float(val_f1),
            "lr":             float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:>2}/{config.MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"train_f1={train_f1:.4f}  val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Always persist the latest checkpoint and history
        save_full_checkpoint(
            config.LAST_CKPT_LOCAL, model, optimizer, scheduler,
            epoch, history, best_macro_f1, best_epoch,
        )
        save_history_json(history, config.HISTORY_JSON_LOCAL)

        if val_f1 > best_macro_f1:
            best_macro_f1  = val_f1
            best_epoch     = epoch
            epochs_no_improve = 0

            save_full_checkpoint(
                config.BEST_CKPT_LOCAL, model, optimizer, scheduler,
                epoch, history, best_macro_f1, best_epoch,
            )
            torch.save(model.state_dict(), config.BEST_WEIGHTS_LOCAL)
            print("    >>> New best model saved!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print("    >>> Early stopping triggered.")
            break

    # ----------------------------------------------------------
    # 10. Summary
    # ----------------------------------------------------------
    total_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("Training Finished")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Best F1    : {best_macro_f1:.4f}")
    print(f"  Duration   : {total_minutes:.1f} min")
    print("=" * 60)

    # ----------------------------------------------------------
    # 11. Plots
    # ----------------------------------------------------------
    plot_and_save_history(history, config.LOCAL_FIG_ROOT, best_epoch, best_macro_f1)

    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    # The if-guard prevents DataLoader worker processes from
    # re-executing the training code on Windows.
    main()
