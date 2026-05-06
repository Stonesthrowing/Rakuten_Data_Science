# ============================================================
# evaluate.py
# Standalone evaluation script for Model I7.
# Loads the best saved weights, runs inference on the
# validation set, and produces:
#   - Classification report (accuracy, macro F1, weighted F1)
#   - Normalised confusion matrix (PNG)
# ============================================================

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

import notebooks.image_modeling.restNet50.Model_I7_ResNet50_ModerateAug_Full.config as config
from notebooks.image_modeling.restNet50.Model_I7_ResNet50_ModerateAug_Full.dataset import RakutenImageDataset
from notebooks.image_modeling.restNet50.Model_I7_ResNet50_ModerateAug_Full.utils import evaluate_model


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run      : {config.RUN_NAME}")
    print(f"Device   : {device}")
    print(f"Weights  : {config.BEST_WEIGHTS_LOCAL}")

    # ----------------------------------------------------------
    # 1. Label mapping
    # ----------------------------------------------------------
    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    id2label     = {v: str(k) for k, v in label2id.items()}
    target_names = [id2label[i] for i in range(len(id2label))]

    # ----------------------------------------------------------
    # 2. Validation dataset & loader
    # ----------------------------------------------------------
    val_df = pd.read_csv(config.SPLIT_DIR / "val_split.csv")
    val_df["label_id"] = val_df["prdtypecode"].map(label2id)
    val_df["image_path_local"] = val_df.apply(
        lambda r: str(config.LOCAL_IMAGE_TRAIN_DIR / f"image_{r['imageid']}_product_{r['productid']}.jpg"),
        axis=1,
    )

    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_loader = DataLoader(
        RakutenImageDataset(val_df, transform=val_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # ----------------------------------------------------------
    # 3. Rebuild model & load best weights
    #    (Full unfreeze – must match train.py exactly)
    # ----------------------------------------------------------
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(config.DROPOUT),
        nn.Linear(model.fc.in_features, len(label2id)),
    )

    model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))
    model = model.to(device)

    # ----------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------
    metrics = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        target_names=target_names,
        fig_dir=config.LOCAL_FIG_ROOT,
        model_name=config.RUN_NAME,
    )

    print(f"\nSummary  |  Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}  |  "
          f"Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
