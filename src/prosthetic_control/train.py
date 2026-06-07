"""Training utilities for baseline and neural models."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .datasets import WindowDataset
from .models import SequenceMLP, TemporalConvNet


def train_random_forest(x_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest baseline on engineered features."""
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def train_torch_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "mlp",
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> tuple[torch.nn.Module, LabelEncoder]:
    """Train a lightweight neural classifier on raw time windows."""
    torch.manual_seed(seed)
    encoder = LabelEncoder().fit(y_train)
    dataset = WindowDataset(x_train, y_train, encoder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_classes = len(encoder.classes_)
    window_size = x_train.shape[1]
    n_channels = x_train.shape[2]

    if model_type == "cnn":
        model = TemporalConvNet(n_channels=n_channels, n_classes=n_classes)
    else:
        model = SequenceMLP(window_size=window_size, n_channels=n_channels, n_classes=n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model, encoder


def predict_torch(model: torch.nn.Module, encoder: LabelEncoder, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Predict labels for raw windows using a trained neural classifier."""
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            logits = model(xb)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    encoded = np.concatenate(predictions)
    return encoder.inverse_transform(encoded)
