# ============================================================
# model.py
# Fusion model architecture:
#   - IntermediateFusionHead  : lightweight MLP over 1792d features
#   - MultimodalGradCAMWrapper: wraps image backbone + fusion head
#                               for Grad-CAM visualisation
# ============================================================

import torch
import torch.nn as nn

import config


class IntermediateFusionHead(nn.Module):
    """
    Lightweight MLP classification head for intermediate fusion.

    Input  : concatenated image (1024d) + text (768d) features → 1792d
    Output : class logits (num_classes)

    Architecture (V2 – regularised):
        Linear(1792 → 256) → BN → ReLU → Dropout(0.5)
        Linear(256  → 128) → BN → ReLU → Dropout(0.4)
        Linear(128  → num_classes)
    """

    def __init__(
        self,
        input_dim:   int = config.INPUT_DIM,
        hidden_dim_1: int = config.HIDDEN_DIM_1,
        hidden_dim_2: int = config.HIDDEN_DIM_2,
        num_classes: int = config.NUM_CLASSES,
        dropout_1:   float = config.DROPOUT_1,
        dropout_2:   float = config.DROPOUT_2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,   hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),

            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalGradCAMWrapper(nn.Module):
    """
    Wraps a ConvNeXt image backbone and the IntermediateFusionHead
    so that Grad-CAM gradients flow through the image model only.

    The text feature vector is registered as a fixed buffer so it
    does not participate in the gradient computation.

    Args:
        visual_model         : ConvNeXt feature extractor (num_classes=0).
        fusion_head          : Trained IntermediateFusionHead.
        static_text_features : Text feature vector for one sample, shape (768,).
    """

    def __init__(self, visual_model, fusion_head, static_text_features):
        super().__init__()
        self.visual_model = visual_model
        self.fusion_head  = fusion_head
        self.register_buffer(
            "text_feat",
            torch.tensor(static_text_features, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_feats = self.visual_model(x)                         # (B, 1024)
        t_feat    = self.text_feat.repeat(img_feats.size(0), 1)  # (B,  768)
        combined  = torch.cat([img_feats, t_feat], dim=1)        # (B, 1792)
        return self.fusion_head(combined)
