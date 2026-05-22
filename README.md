# Italtensor

Italtensor is a local desktop workbench for small binary-classification experiments. It is designed for quick iteration: load or create a dataset, run a model, inspect validation metrics, export a report, and save a reusable model or dataset preset.

The app runs without TensorFlow by default. When TensorFlow is unavailable, Italtensor uses a NumPy logistic trainer with optional nonlinear feature maps. TensorFlow/Keras remains available as an optional backend for environments that can install it.

## What It Does

- Desktop GUI built with PySimpleGUI.
- Manual JSON sample entry and CSV import.
- Built-in dataset presets for quick experiments:
  - Linear blobs
  - XOR pattern
  - Imbalanced blobs
  - Signal plus noise
  - Concentric rings
  - Two moons
  - Rare event signal
- Save and import reusable dataset presets.
- Train once or run random-search experiments.
- No-TensorFlow fallback trainer with linear, quadratic, and random Fourier feature maps.
- Validation-tuned decision threshold instead of a fixed `0.5` cutoff.
- Class weighting for imbalanced binary datasets.
- Feature standardization fitted only on the training split.
- Permutation-style feature importance on validation data.
- JSON or Markdown experiment reports.
- Model save/load:
  - `.italtensor-model.json` for NumPy fallback models
  - `.keras` for TensorFlow/Keras models when TensorFlow is installed

## Install

Base install:

```powershell
python -m pip install -r requirements.txt
```

Optional TensorFlow backend:

```powershell
python -m pip install -r requirements-tensorflow.txt
```

TensorFlow is intentionally optional because the Windows wheel is large. The base app can train and predict without it.

## Run

```powershell
python main.py
```

## Quick Start

1. Choose a built-in dataset preset.
2. Keep `Map` as `rff` or choose `quadratic` for XOR-style patterns.
3. Click `Load preset`.
4. Click `Train once` or `Run auto experiments`.
5. Enter a prediction vector such as `[0.2, 0.8]`.
6. Click `Predict`.
7. Optionally save the model, export a report, or save the dataset as a preset.

## Input Formats

Manual training sample:

```json
[[0.1, 0.2, 0.3], 1]
```

Prediction vector:

```json
[0.1, 0.2, 0.3]
```

CSV datasets use the final column as the binary label:

```csv
f1,f2,label
0.1,0.2,0
0.9,0.7,1
```

Labels must be `0` or `1`. Feature values must be finite numbers. All rows must have the same number of features.

## Dataset Presets

Preset files wrap the normal dataset shape with metadata:

```json
{
  "kind": "italtensor.dataset_preset",
  "schema_version": 1,
  "name": "My preset",
  "description": "Reusable local dataset",
  "training_defaults": {
    "epochs": 50,
    "batch_size": 16,
    "trials": 8
  },
  "prediction_examples": [],
  "dataset": {
    "input_dim": 2,
    "samples": [
      {"features": [0.1, 0.2], "label": 0},
      {"features": [0.9, 0.7], "label": 1}
    ]
  }
}
```

Use `Save as preset` to turn the current dataset into a reusable preset.

## Experiment Notes

The NumPy fallback is intentionally lightweight. It trains a logistic classifier, but can expand features before fitting:

- `linear`: fastest, best for simple boundaries.
- `quadratic`: adds squared and interaction terms; useful for XOR-like structure.
- `rff`: random Fourier features inspired by random kitchen sinks, useful for nonlinear boundaries without adding heavy dependencies.

Feature maps are used by the NumPy fallback backend. If TensorFlow is installed and active, Keras models ignore the feature-map selector.

Auto experiments search model settings and feature maps, then rank runs by validation F1, accuracy, and validation loss.

## Testing

```powershell
python -m pytest -q -rs
pytest -q -p no:cacheprovider
```

TensorFlow-specific tests skip when TensorFlow is not installed.

## Current Limits

- Italtensor is for local small-to-medium binary classification experiments, not production ML serving.
- The fallback trainer is not a neural network; nonlinear feature maps make it stronger but still lightweight.
- Existing `.keras` files require TensorFlow to load.
- Fallback model metadata sidecars currently use `.italtensor-model.json.json`.
- Reports are richest when the dataset is loaded in the same session as the model; model-only reports cannot reconstruct class counts.
- Validation metrics need enough examples from both classes. The app requires at least two samples per class for train/validation splitting.

## Research And Design Anchors

- Polynomial feature expansion follows the same idea exposed by scikit-learn `PolynomialFeatures`.
- Random Fourier features are based on Rahimi and Recht's random kitchen sinks work for approximating kernel methods.
- Brier/log-loss style probability evaluation and threshold tuning are standard classification-evaluation tools; Italtensor currently exposes threshold-tuned metrics and report export, with more calibration diagnostics a natural next addition.
