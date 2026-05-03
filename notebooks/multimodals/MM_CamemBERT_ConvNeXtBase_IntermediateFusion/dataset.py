# ============================================================
# dataset.py
# Fusion Dataset – loads pre-extracted image and text features
# and concatenates them into a single feature vector.
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class RakutenFusionDataset(Dataset):
    """
    PyTorch Dataset for intermediate fusion over pre-extracted features.

    Concatenates ConvNeXt image features (1024d) and CamemBERT text
    features (768d) into a single 1792d vector per sample.

    Args:
        img_feat_path : Path to .npy file with image features (N, 1024).
        txt_feat_path : Path to .npy file with text features  (N,  768).
        df            : DataFrame with 'prdtypecode' column.
        label2id      : Dict mapping prdtypecode → integer label.
    """

    def __init__(self, img_feat_path, txt_feat_path, df, label2id):
        self.img_features = np.load(img_feat_path)   # (N, 1024)
        self.txt_features = np.load(txt_feat_path)   # (N,  768)

        assert len(self.img_features) == len(self.txt_features), (
            f"Feature count mismatch: image={len(self.img_features)}, "
            f"text={len(self.txt_features)}. "
            "Ensure both were exported from the same split."
        )

        mapping = {str(k): v for k, v in label2id.items()}
        df = df.copy()
        df["label_id"] = df["prdtypecode"].astype(str).map(mapping)
        self.labels = df["label_id"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        combined = np.concatenate(
            [self.img_features[idx], self.txt_features[idx]], axis=0
        )
        return (
            torch.tensor(combined, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
