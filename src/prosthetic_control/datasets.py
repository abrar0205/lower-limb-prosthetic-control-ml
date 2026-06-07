"""Dataset helpers for PyTorch sequence models."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """PyTorch dataset for windowed multimodal sensor sequences."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, label_encoder: LabelEncoder | None = None):
        self.windows = windows.astype(np.float32)
        if label_encoder is None:
            label_encoder = LabelEncoder().fit(labels)
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(labels).astype(np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.windows[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


def encode_labels(train_labels: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode string labels into integer targets."""
    encoder = LabelEncoder().fit(train_labels)
    return encoder.transform(train_labels), encoder.transform(test_labels), encoder
