# ============================================================
# dataset.py
# Rakuten Image Dataset – PyTorch Dataset wrapper
# ============================================================

import torch
from PIL import Image
from torch.utils.data import Dataset


class RakutenImageDataset(Dataset):
    """
    PyTorch Dataset for the Rakuten product image classification task.

    Args:
        df         : DataFrame with at least image path and label columns.
        transform  : torchvision transform pipeline (optional).
        path_col   : Column name that holds the absolute image path string.
        label_col  : Column name that holds the integer class label.
    """

    def __init__(
        self,
        df,
        transform=None,
        path_col: str = "image_path_local",
        label_col: str = "label_id",
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, self.path_col]
        label = int(self.df.loc[idx, self.label_col])

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print(f"[WARNING] Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))

        return image, label
