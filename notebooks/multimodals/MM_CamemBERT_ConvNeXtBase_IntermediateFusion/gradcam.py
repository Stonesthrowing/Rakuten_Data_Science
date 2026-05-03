# ============================================================
# gradcam.py
# Grad-CAM visualisation for a single validation sample.
#
# Prerequisites:
#   pip install grad-cam timm
#
# Workflow:
#   1. Run analyze_errors.py to find interesting indices.
#   2. Set GRADCAM_TARGET_INDEX in config.py.
#   3. Run this script.
#
# The script shows why the model predicted the wrong class by
# highlighting which image regions drove that decision.
# ============================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm import create_model
from torchvision import transforms

import config
from model import IntermediateFusionHead, MultimodalGradCAMWrapper


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx    = config.GRADCAM_TARGET_INDEX

    print(f"Run          : {config.RUN_NAME}")
    print(f"Target index : {idx}")

    # ----------------------------------------------------------
    # 1. Load val metadata & resolve predicted label
    # ----------------------------------------------------------
    val_df = pd.read_csv(config.SPLIT_DIR / "val_split.csv").reset_index(drop=True)

    with open(config.SPLIT_DIR / "label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {int(k): int(v) for k, v in label2id.items()}
    mapping  = {str(k): v for k, v in label2id.items()}

    val_df["true_label"] = val_df["prdtypecode"].astype(str).map(mapping)

    # Run quick inference to get the predicted label for this sample
    img_feats = np.load(config.VAL_FEATURES_IMAGE)
    txt_feats = np.load(config.VAL_FEATURES_TEXT)

    fusion_model = IntermediateFusionHead().to(device)
    fusion_model.load_state_dict(
        torch.load(config.BEST_WEIGHTS_LOCAL, map_location=device)
    )
    fusion_model.eval()

    x_single = torch.tensor(
        np.concatenate([img_feats[idx], txt_feats[idx]], axis=0),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = fusion_model(x_single)
        pred_label = int(torch.argmax(logits, dim=1).item())

    row = val_df.loc[idx]
    true_label = int(row["true_label"])

    print(f"True label   : {true_label}")
    print(f"Pred label   : {pred_label}")

    # ----------------------------------------------------------
    # 2. Load image
    # ----------------------------------------------------------
    img_filename = f"image_{row['imageid']}_product_{row['productid']}.jpg"
    img_path     = config.IMAGE_DIR / img_filename

    raw_img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    # ----------------------------------------------------------
    # 3. Build Grad-CAM wrapper
    # ----------------------------------------------------------
    # ConvNeXt-Base as pure feature extractor (num_classes=0)
    visual_model = create_model(
        "convnext_base", pretrained=False, num_classes=0
    ).to(device)

    # Load trained image model weights (optional – improves CAM quality)
    if config.IMAGE_MODEL_WEIGHTS.exists():
        state = torch.load(config.IMAGE_MODEL_WEIGHTS, map_location=device)
        # Remove classifier keys if present
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        visual_model.load_state_dict(state, strict=False)
        print(f"Image model weights loaded from: {config.IMAGE_MODEL_WEIGHTS}")
    else:
        print("Image model weights not found – using random weights for CAM.")

    txt_vec      = txt_feats[idx].reshape(1, -1)
    wrapped      = MultimodalGradCAMWrapper(visual_model, fusion_model, txt_vec).to(device)
    wrapped.eval()

    # ----------------------------------------------------------
    # 4. Compute Grad-CAM
    # ----------------------------------------------------------
    target_layers = [visual_model.stages[-1].blocks[-1]]
    cam           = GradCAM(model=wrapped, target_layers=target_layers)
    targets       = [ClassifierOutputTarget(pred_label)]

    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    # ----------------------------------------------------------
    # 5. Visualise
    # ----------------------------------------------------------
    rgb_img   = np.array(raw_img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Grad-CAM  |  True: {true_label}  |  Predicted: {pred_label}",
        fontsize=13,
    )
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM (Predicted class: {pred_label})")
    axes[1].axis("off")

    plt.tight_layout()

    out_path = config.LOCAL_FIG_ROOT / f"gradcam_idx{idx}.png"
    config.LOCAL_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Grad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()
