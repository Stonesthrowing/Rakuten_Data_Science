# ============================================================
# evaluate.py
# Standalone evaluation for MM_CamemBERT_ConvNeXtBase_IntermediateFusion.
# Loads best weights, evaluates on val set, saves report + confusion matrix.
# ============================================================

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import RakutenFusionDataset
from model import IntermediateFusionHead
from utils import evaluate_model


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run     : {config.RUN_NAME}")
    print(f"Device  : {device}")
    print(f"Weights : {config.BEST_WEIGHTS_LOCAL}")

    # ----------------------------------------------------------
    # 1. Label mapping
    # ----------------------------------------------------------
    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    id2label     = {v: str(k) for k, v in label2id.items()}
    target_names = [id2label[i] for i in range(len(id2label))]

    # ----------------------------------------------------------
    # 2. Val dataset & loader
    # ----------------------------------------------------------
    val_df = pd.read_csv(config.SPLIT_DIR / "val_split.csv")

    val_ds = RakutenFusionDataset(
        config.VAL_FEATURES_IMAGE, config.VAL_FEATURES_TEXT, val_df, label2id
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # ----------------------------------------------------------
    # 3. Load model
    # ----------------------------------------------------------
    model = IntermediateFusionHead()
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

    print(
        f"\nSummary  |  Accuracy: {metrics['accuracy']:.4f}  |  "
        f"Macro F1: {metrics['macro_f1']:.4f}  |  "
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
