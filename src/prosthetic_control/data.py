"""Synthetic multimodal EMG/IMU data generation.

The generator creates reproducible lower-limb movement trials with EMG-like
activation envelopes and IMU-like acceleration/gyro channels. The goal is to
support ML workflow demonstration without depending on private or clinical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MOVEMENT_CLASSES = ["level_walk", "stair_ascent", "stair_descent", "sit_to_stand"]
GAIT_PHASES = ["stance", "swing", "transition"]
EMG_CHANNELS = ["emg_tibialis", "emg_gastrocnemius", "emg_quadriceps", "emg_hamstrings"]
IMU_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
SENSOR_CHANNELS = EMG_CHANNELS + IMU_CHANNELS


@dataclass(frozen=True)
class DatasetConfig:
    n_trials_per_class: int = 18
    samples_per_trial: int = 240
    sampling_rate_hz: int = 100
    noise_level: float = 0.08
    random_state: int = 42


def _phase_from_cycle(cycle: np.ndarray) -> np.ndarray:
    phase = np.empty(cycle.shape, dtype=object)
    phase[cycle < 0.55] = "stance"
    phase[(cycle >= 0.55) & (cycle < 0.90)] = "swing"
    phase[cycle >= 0.90] = "transition"
    return phase


def _activation(cycle: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((cycle - center) / width) ** 2)


def _movement_profile(movement: str) -> dict[str, float]:
    profiles = {
        "level_walk": {"freq": 1.00, "amp": 1.00, "tilt": 0.10},
        "stair_ascent": {"freq": 0.85, "amp": 1.25, "tilt": 0.35},
        "stair_descent": {"freq": 0.90, "amp": 1.15, "tilt": -0.30},
        "sit_to_stand": {"freq": 0.45, "amp": 1.45, "tilt": 0.60},
    }
    return profiles[movement]


def generate_trial(
    movement: str,
    trial_id: int,
    config: DatasetConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one synthetic multimodal trial."""
    n = config.samples_per_trial
    t = np.arange(n) / config.sampling_rate_hz
    profile = _movement_profile(movement)
    frequency = profile["freq"] * rng.normal(1.0, 0.04)
    cycle = (t * frequency + rng.uniform(0, 1)) % 1.0
    phase = _phase_from_cycle(cycle)
    amp = profile["amp"]
    noise = config.noise_level

    emg_tibialis = _activation(cycle, 0.68, 0.10, 0.9 * amp) + _activation(cycle, 0.05, 0.08, 0.25 * amp)
    emg_gastro = _activation(cycle, 0.35, 0.14, 1.0 * amp)
    emg_quad = _activation(cycle, 0.10, 0.16, 0.8 * amp) + _activation(cycle, 0.82, 0.08, 0.35 * amp)
    emg_ham = _activation(cycle, 0.92, 0.11, 0.75 * amp)

    emg = np.vstack([emg_tibialis, emg_gastro, emg_quad, emg_ham]).T
    emg += rng.normal(0, noise, emg.shape)
    emg = np.clip(emg, 0, None)

    base_sin = np.sin(2 * np.pi * cycle)
    base_cos = np.cos(2 * np.pi * cycle)
    tilt = profile["tilt"]
    imu = np.vstack(
        [
            amp * base_sin + tilt,
            0.6 * amp * base_cos,
            0.25 * amp * np.sin(4 * np.pi * cycle) + 0.1,
            0.8 * amp * base_cos,
            0.5 * amp * base_sin + tilt,
            0.35 * amp * np.sin(4 * np.pi * cycle),
        ]
    ).T
    imu += rng.normal(0, noise * 0.7, imu.shape)

    data = pd.DataFrame(np.hstack([emg, imu]), columns=SENSOR_CHANNELS)
    data.insert(0, "time_s", t)
    data.insert(0, "sample", np.arange(n))
    data.insert(0, "gait_phase", phase)
    data.insert(0, "movement", movement)
    data.insert(0, "trial_id", trial_id)
    return data


def generate_dataset(config: DatasetConfig | None = None) -> pd.DataFrame:
    """Generate a complete synthetic dataset."""
    config = config or DatasetConfig()
    rng = np.random.default_rng(config.random_state)
    trials = []
    trial_id = 0
    for movement in MOVEMENT_CLASSES:
        for _ in range(config.n_trials_per_class):
            trials.append(generate_trial(movement, trial_id, config, rng))
            trial_id += 1
    return pd.concat(trials, ignore_index=True)
