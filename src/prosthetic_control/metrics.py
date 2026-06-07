"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def nrmsd(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    value = np.sqrt(np.mean((reference - prediction) ** 2))
    scale = np.max(reference) - np.min(reference)
    if scale < 1e-12:
        scale = np.std(reference) + 1e-12
    return float(value / scale)


def temporal_consistency(predictions: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    if len(predictions) < 2:
        return 1.0
    return float(np.mean(predictions[1:] == predictions[:-1]))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "temporal_consistency": temporal_consistency(y_pred),
    }


def score_drop(clean_score: float, noisy_score: float) -> float:
    return float(clean_score - noisy_score)
