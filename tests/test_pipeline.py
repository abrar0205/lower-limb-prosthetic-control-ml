import numpy as np

from prosthetic_control.data import DatasetConfig, generate_dataset
from prosthetic_control.features import extract_feature_matrix
from prosthetic_control.preprocessing import create_windows, normalize_channels, split_by_trial
from prosthetic_control.metrics import nrmsd, temporal_consistency


def test_dataset_generation():
    config = DatasetConfig(n_trials_per_class=2, samples_per_trial=80, random_state=7)
    df = generate_dataset(config)
    assert not df.empty
    assert "movement" in df.columns
    assert "emg_tibialis" in df.columns
    assert "acc_x" in df.columns


def test_windowing_and_features():
    config = DatasetConfig(n_trials_per_class=2, samples_per_trial=80, random_state=7)
    df = normalize_channels(generate_dataset(config))
    windows, labels, gait, trials = create_windows(df, window_size=24, step_size=12)
    assert windows.ndim == 3
    assert len(labels) == len(windows)
    assert len(gait) == len(windows)
    assert len(trials) == len(windows)

    features = extract_feature_matrix(windows[:3])
    assert not features.empty
    assert any(column.endswith("_rms") for column in features.columns)


def test_split_and_metrics():
    config = DatasetConfig(n_trials_per_class=2, samples_per_trial=80, random_state=7)
    df = normalize_channels(generate_dataset(config))
    windows, labels, _, trials = create_windows(df, window_size=24, step_size=12)
    x_train, x_test, y_train, y_test = split_by_trial(windows, labels, trials)
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)

    value = nrmsd(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert value == 0
    assert 0 <= temporal_consistency(np.array([1, 1, 2, 2])) <= 1
