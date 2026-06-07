"""Machine learning models for prosthetic control experiments."""

from __future__ import annotations

import torch
from torch import nn


class SequenceMLP(nn.Module):
    """Compact neural model for windowed EMG/IMU sequences.

    The model flattens a time window and learns temporal patterns through dense
    layers. It is intentionally lightweight for fast portfolio demos.
    """

    def __init__(self, window_size: int, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = window_size * n_channels
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalConvNet(nn.Module):
    """Small 1D CNN for temporal sensor pattern recognition."""

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: batch x time x channels. Conv1d expects batch x channels x time.
        x = x.transpose(1, 2)
        return self.net(x)
