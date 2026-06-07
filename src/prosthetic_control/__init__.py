"""Lower-limb prosthetic control ML package."""

from .data import generate_dataset
from .preprocessing import create_windows, normalize_channels
from .features import extract_feature_matrix
from .metrics import nrmsd, temporal_consistency

__all__ = [
    "generate_dataset",
    "create_windows",
    "normalize_channels",
    "extract_feature_matrix",
    "nrmsd",
    "temporal_consistency",
]
