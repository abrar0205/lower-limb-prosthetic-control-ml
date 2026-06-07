# Lower-Limb Prosthetic Control ML

Machine learning project for multimodal EMG and IMU time-series modelling in wearable prosthetic control.

This repository demonstrates an end-to-end workflow for movement prediction, gait-phase segmentation, user-intent recognition, multimodal sensor fusion, and model evaluation using reproducible synthetic signal data.

## Highlights

- Synthetic EMG and IMU time-series generation
- Sliding-window preprocessing and feature extraction
- Gait-phase and movement-intent classification
- EMG-only, IMU-only, and fused sensor models
- Baseline Random Forest model and PyTorch MLP model
- Evaluation with accuracy, macro F1, NRMSD, temporal consistency, and robustness under noise

## Project structure

```text
examples/run_pipeline.py
src/prosthetic_control/data.py
src/prosthetic_control/preprocessing.py
src/prosthetic_control/features.py
src/prosthetic_control/datasets.py
src/prosthetic_control/models.py
src/prosthetic_control/metrics.py
src/prosthetic_control/train.py
src/prosthetic_control/evaluate.py
tests/test_pipeline.py
```

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
python examples/run_pipeline.py
```

The script generates synthetic EMG/IMU data, trains models, evaluates fusion strategies, and writes summary results to `outputs/`.

## Workflow

1. Generate synthetic EMG/IMU trials for lower-limb movement classes
2. Normalize sensor channels and create sliding windows
3. Extract time-domain and frequency-domain features
4. Train baseline and neural-network models
5. Compare EMG-only, IMU-only, and multimodal-fusion performance
6. Evaluate temporal stability and robustness under noise

## Academic context

This project reflects seminar work on hybrid neuromusculoskeletal modelling for wearable prosthetic control. It focuses on multimodal time-series analysis, movement prediction, gait understanding, user-intent recognition, feature extraction, model evaluation, and practical system integration.

## Note

The data is synthetic and generated locally for reproducibility.
