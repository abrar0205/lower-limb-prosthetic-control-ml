"""Feature extraction for EMG and IMU windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EMG_CHANNELS, IMU_CHANNELS, SENSOR_CHANNELS


def waveform_length(window: np.ndarray) -> np.ndarray:
    """Compute waveform length per channel."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def zero_crossings(window: np.ndarray) -> np.ndarray:
    """Estimate zero crossings per channel."""
    signs = np.sign(window)
    return np.sum(np.diff(signs, axis=0) != 0, axis=0)


def spectral_energy(window: np.ndarray) -> np.ndarray:
    """Compute normalized spectral energy per channel."""
    spectrum = np.fft.rfft(window, axis=0)
    energy = np.sum(np.abs(spectrum) ** 2, axis=0)
    return energy / (window.shape[0] + 1e-8)


def extract_window_features(window: np.ndarray, channel_names: list[str] | None = None) -> dict[str, float]:
    """Extract time-domain and frequency-domain features from one window."""
    channel_names = channel_names or SENSOR_CHANNELS
    features: dict[str, float] = {}

    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0))
    mav = np.mean(np.abs(window), axis=0)
    wl = waveform_length(window)
    zc = zero_crossings(window)
    se = spectral_energy(window)
    dynamic_range = np.max(window, axis=0) - np.min(window, axis=0)

    for idx, channel in enumerate(channel_names):
        features[f"{channel}_mean"] = float(mean[idx])
        features[f"{channel}_std"] = float(std[idx])
        features[f"{channel}_rms"] = float(rms[idx])
        features[f"{channel}_mav"] = float(mav[idx])
        features[f"{channel}_waveform_length"] = float(wl[idx])
        features[f"{channel}_zero_crossings"] = float(zc[idx])
        features[f"{channel}_spectral_energy"] = float(se[idx])
        features[f"{channel}_range"] = float(dynamic_range[idx])

    emg_indices = [channel_names.index(ch) for ch in EMG_CHANNELS if ch in channel_names]
    imu_indices = [channel_names.index(ch) for ch in IMU_CHANNELS if ch in channel_names]

    if emg_indices:
        features["emg_global_rms"] = float(np.mean(rms[emg_indices]))
        features["emg_total_activation"] = float(np.sum(mav[emg_indices]))

    if imu_indices:
        features["imu_global_std"] = float(np.mean(std[imu_indices]))
        features["imu_dynamic_range"] = float(np.mean(dynamic_range[imu_indices]))

    return features


def extract_feature_matrix(windows: np.ndarray, channel_names: list[str] | None = None) -> pd.DataFrame:
    """Extract a feature table from multiple windows."""
    rows = [extract_window_features(window, channel_names) for window in windows]
    return pd.DataFrame(rows)
