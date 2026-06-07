"""Evaluation helpers for multimodal prosthetic-control models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from .metrics import classification_metrics, nrmsd, score_drop


def add_sensor_noise(x: np.ndarray, noise_std: float = 0.15, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to sensor windows for robustness testing."""
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return classification and stability metrics."""
    return classification_metrics(y_true, y_pred)


def compare_clean_noisy(y_true: np.ndarray, clean_pred: np.ndarray, noisy_pred: np.ndarray) -> dict[str, float]:
    """Compare model behavior under clean and noisy input conditions."""
    clean = classification_metrics(y_true, clean_pred)
    noisy = classification_metrics(y_true, noisy_pred)
    return {
        "clean_accuracy": clean["accuracy"],
        "noisy_accuracy": noisy["accuracy"],
        "accuracy_drop": score_drop(clean["accuracy"], noisy["accuracy"]),
        "clean_macro_f1": clean["macro_f1"],
        "noisy_macro_f1": noisy["macro_f1"],
        "macro_f1_drop": score_drop(clean["macro_f1"], noisy["macro_f1"]),
    }


def trajectory_similarity(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Evaluate similarity of continuous trajectories or activation patterns."""
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    corr = np.corrcoef(reference.ravel(), predicted.ravel())[0, 1]
    return {"nrmsd": nrmsd(reference, predicted), "correlation": float(corr)}


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Return a confusion matrix with fixed label order."""
    return confusion_matrix(y_true, y_pred, labels=labels)
