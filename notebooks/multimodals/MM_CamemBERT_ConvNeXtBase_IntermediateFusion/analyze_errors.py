# ============================================================
# analyze_errors.py
# Find the top-N validation samples where the fusion model is
# most confidently wrong.
#
# Output: printed table + CSV saved to LOCAL_OUTPUT_ROOT.
# Use the row index from the output to set GRADCAM_TARGET_INDEX
# in config.py before running gradcam.py.
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from model import IntermediateFusionHead


def main(top_n: int = 5) -> pd.DataFrame:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run     : {config.RUN_NAME}")
    print(f"Weights : {config.BEST_WEIGHTS_LOCAL}")

    # ----------------------------------------------------------
    # 1. Load features & labels
    # ----------------------------------------------------------
    img_feats = np.load(config.VAL_FEATURES_IMAGE)
    txt_feats = np.load(config.VAL_FEATURES_TEXT)

    val_df = pd.read_csv(config.SPLIT_DIR / "val_split.csv").reset_index(drop=True)

    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}

    mapping = {str(k): v for k, v in label2id.items()}
    val_df["true_label"] = val_df["prdtypecode"].astype(str).map(mapping)

    # ----------------------------------------------------------
    # 2. Load model & run inference
    # ----------------------------------------------------------
    model = IntermediateFusionHead()
    model.load_state_dict(torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device))
    model = model.to(device).eval()

    X = torch.tensor(
        np.concatenate([img_feats, txt_feats], axis=1), dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits      = model(X)
        probs       = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    val_df["pred_label"]  = preds.cpu().numpy()
    val_df["confidence"]  = confidences.cpu().numpy()

    # ----------------------------------------------------------
    # 3. Filter mistakes and rank by confidence
    # ----------------------------------------------------------
    mistakes   = val_df[val_df["true_label"] != val_df["pred_label"]].copy()
    top_errors = mistakes.sort_values("confidence", ascending=False).head(top_n)

    cols = ["imageid", "productid", "true_label", "pred_label", "confidence"]
    print(f"\nTop {top_n} high-confidence errors:")
    print(top_errors[cols].to_string(index=True))
    print(
        f"\nTip: set GRADCAM_TARGET_INDEX in config.py to one of the "
        f"indices above, then run gradcam.py."
    )

    # Save to CSV
    out_path = config.LOCAL_OUTPUT_ROOT / f"top_{top_n}_errors.csv"
    config.LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    top_errors[cols].to_csv(out_path, index=True)
    print(f"Saved to: {out_path}")

    return top_errors


if __name__ == "__main__":
    main(top_n=5)
