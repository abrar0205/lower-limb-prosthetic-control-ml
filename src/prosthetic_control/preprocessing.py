"""Preprocessing utilities for multimodal time-series data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SENSOR_CHANNELS


def normalize_channels(df: pd.DataFrame, channels: list[str] | None = None) -> pd.DataFrame:
    """Z-score normalize sensor channels across the dataset."""
    channels = channels or SENSOR_CHANNELS
    out = df.copy()
    for channel in channels:
        mean = out[channel].mean()
        std = out[channel].std() + 1e-8
        out[channel] = (out[channel] - mean) / std
    return out


def majority_label(values: pd.Series) -> str:
    """Return the most frequent label in a window."""
    return values.value_counts().idxmax()


def create_windows(
    df: pd.DataFrame,
    window_size: int = 48,
    step_size: int = 12,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows grouped by trial.

    Returns
    -------
    windows:
        Shape: n_windows x window_size x n_channels
    movement_labels:
        Movement intent label per window
    gait_labels:
        Gait phase label per window
    trial_ids:
        Trial identifier per window
    """
    channels = channels or SENSOR_CHANNELS
    x_windows: list[np.ndarray] = []
    movement_labels: list[str] = []
    gait_labels: list[str] = []
    trial_ids: list[int] = []

    for trial_id, group in df.groupby("trial_id"):
        group = group.sort_values("sample")
        values = group[channels].to_numpy(dtype=np.float32)
        for start in range(0, len(group) - window_size + 1, step_size):
            stop = start + window_size
            window_df = group.iloc[start:stop]
            x_windows.append(values[start:stop])
            movement_labels.append(majority_label(window_df["movement"]))
            gait_labels.append(majority_label(window_df["gait_phase"]))
            trial_ids.append(int(trial_id))

    return (
        np.stack(x_windows),
        np.asarray(movement_labels),
        np.asarray(gait_labels),
        np.asarray(trial_ids),
    )


def split_by_trial(
    x: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split windows by trial IDs to reduce leakage across train/test sets."""
    rng = np.random.default_rng(random_state)
    unique_trials = np.unique(trial_ids)
    rng.shuffle(unique_trials)
    n_test = max(1, int(round(len(unique_trials) * test_fraction)))
    test_trials = set(unique_trials[:n_test])
    test_mask = np.array([trial in test_trials for trial in trial_ids])
    return x[~test_mask], x[test_mask], y[~test_mask], y[test_mask]
