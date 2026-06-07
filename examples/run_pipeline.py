"""Run the full prosthetic-control ML workflow.

Usage:
    python examples/run_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

from prosthetic_control.data import DatasetConfig, EMG_CHANNELS, IMU_CHANNELS, SENSOR_CHANNELS, generate_dataset
from prosthetic_control.evaluate import add_sensor_noise, compare_clean_noisy, summarize_predictions
from prosthetic_control.features import extract_feature_matrix
from prosthetic_control.preprocessing import create_windows, normalize_channels, split_by_trial
from prosthetic_control.train import predict_torch, train_random_forest, train_torch_classifier


def plot_signal_preview(df: pd.DataFrame, output_dir: Path) -> None:
    trial = df[df["trial_id"] == df["trial_id"].iloc[0]]
    plt.figure(figsize=(10, 5))
    plt.plot(trial["time_s"], trial["emg_tibialis"], label="EMG tibialis")
    plt.plot(trial["time_s"], trial["acc_x"], label="IMU acc_x")
    plt.title("Synthetic EMG/IMU trial preview")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "signal_preview.png", dpi=160)
    plt.close()


def run_modality_experiment(windows, labels, trials, channel_slice, name: str) -> dict[str, float]:
    x_modality = windows[:, :, channel_slice]
    x_train, x_test, y_train, y_test = split_by_trial(x_modality, labels, trials)
    features_train = extract_feature_matrix(x_train)
    features_test = extract_feature_matrix(x_test)
    rf = train_random_forest(features_train, y_train)
    predictions = rf.predict(features_test)
    metrics = summarize_predictions(y_test, predictions)
    return {f"{name}_{key}": value for key, value in metrics.items()}


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    config = DatasetConfig(n_trials_per_class=20, samples_per_trial=260, noise_level=0.08)
    df = generate_dataset(config)
    df = normalize_channels(df)
    df.to_csv(output_dir / "synthetic_emg_imu_dataset.csv", index=False)
    plot_signal_preview(df, output_dir)

    windows, movement_labels, gait_labels, trial_ids = create_windows(df, window_size=52, step_size=13)

    x_train, x_test, y_train, y_test = split_by_trial(windows, movement_labels, trial_ids)

    features_train = extract_feature_matrix(x_train)
    features_test = extract_feature_matrix(x_test)

    rf = train_random_forest(features_train, y_train)
    rf_pred = rf.predict(features_test)
    rf_metrics = summarize_predictions(y_test, rf_pred)

    neural_model, encoder = train_torch_classifier(x_train, y_train, model_type="cnn", epochs=6)
    nn_pred = predict_torch(neural_model, encoder, x_test)
    nn_metrics = summarize_predictions(y_test, nn_pred)

    noisy_x_test = add_sensor_noise(x_test, noise_std=0.20)
    noisy_pred = predict_torch(neural_model, encoder, noisy_x_test)
    robustness = compare_clean_noisy(y_test, nn_pred, noisy_pred)

    emg_results = run_modality_experiment(windows, movement_labels, trial_ids, slice(0, len(EMG_CHANNELS)), "emg_only")
    imu_results = run_modality_experiment(windows, movement_labels, trial_ids, slice(len(EMG_CHANNELS), len(SENSOR_CHANNELS)), "imu_only")

    results = {
        **{f"random_forest_{k}": v for k, v in rf_metrics.items()},
        **{f"temporal_cnn_{k}": v for k, v in nn_metrics.items()},
        **robustness,
        **emg_results,
        **imu_results,
    }

    pd.DataFrame([results]).to_csv(output_dir / "metrics_summary.csv", index=False)

    report = classification_report(y_test, nn_pred)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print("Pipeline complete.")
    print(pd.Series(results).round(3).to_string())
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
