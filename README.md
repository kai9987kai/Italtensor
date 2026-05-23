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
  - Overlapping margin
  - Noisy labels
  - Sparse interaction signal
  - Deployment drift probe
- Save and import reusable dataset presets.
- Train once or run random-search experiments.
- Batch prediction CSV export for scoring unlabeled rows.
- Dataset audit for imbalance, duplicates, label conflicts, constant features, and correlated features.
- No-TensorFlow fallback trainer with linear, quadratic, and random Fourier feature maps.
- Validation-tuned decision threshold plus fixed-`0.5` baseline metrics.
- Class weighting for imbalanced binary datasets.
- Feature standardization fitted only on the training split.
- Permutation-style feature importance on validation data.
- Probability diagnostics including Brier score, log loss, calibration error, ROC-AUC, and average precision.
- Split-conformal and **APS** (Adaptive Prediction Sets) uncertainty diagnostics for abstention experiments.
- **MPS tensor-chain** binary classifier (`backend=mps`): ordered features as sites, bond dimension chi, soft site embeddings.
- Dataset **audit** (imbalance, duplicates, constant/correlated features) in the workbench and reports.
- **Learning curve** diagnostics (F1 vs. training fraction).
- Trial-history export for comparing auto-experiment runs.
- Explicit training backend control: `auto`, `numpy`, `mps`, or `keras` (when TensorFlow is installed).
- Multi-backend run queue: train NumPy, MPS, and Keras models in one sweep and store each in the registry.
- Model panel communication: per-model votes, consensus fusion (mean, median, vote, weighted, stacking), and disagreement scores.
- Persistable model registry (`.json`) for NumPy models and ensembles across sessions.
- JSON or Markdown experiment reports.
- Model save/load:
  - `.italtensor-model.json` for NumPy fallback models
  - `.italtensor-mps.json` for MPS chain models
  - `.italtensor-meta.json` sidecar metadata (no double `.json` suffix)
  - `.keras` for TensorFlow/Keras models when TensorFlow is installed

## Install

Editable install (recommended):

```powershell
python -m pip install -e .
```

Base install from requirements:

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
2. Keep `Map` as `rff` or choose `quadratic` for XOR-style patterns when using the default NumPy fallback.
3. Click `Load preset`.
4. Click `Train once` or `Run auto experiments`.
5. Enter a prediction vector such as `[0.2, 0.8]`.
6. Click `Predict`.
7. Optionally export batch predictions, save the model, export a report, or save the dataset as a preset.

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

Batch prediction CSV files are unlabeled: every column is treated as a numeric feature and the column count must match the trained model's raw input dimension. The exported CSV adds probability, predicted label, conformal set, uncertainty score, drift score, max absolute z-score, OOD flag, and review priority columns.

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
    "trials": 8,
    "feature_map": "linear"
  },
  "recommended_feature_map": "linear",
  "feature_names": ["x1", "x2"],
  "label_names": {"0": "negative", "1": "positive"},
  "prediction_examples": [
    {"name": "Likely positive", "features": [0.9, 0.7], "expected_label": 1}
  ],
  "dataset": {
    "input_dim": 2,
    "samples": [
      {"features": [0.1, 0.2], "label": 0},
      {"features": [0.9, 0.7], "label": 1}
    ]
  }
}
```

Use `Save as preset` to turn the current dataset into a reusable preset. Built-in and imported presets can carry training defaults, a recommended feature map, feature names, label names, and prediction examples; the GUI applies core training defaults when loading presets.

## Experiment Notes

The NumPy fallback is intentionally lightweight. It trains a logistic classifier, but can expand features before fitting:

- `linear`: fastest, best for simple boundaries.
- `quadratic`: adds squared and interaction terms; useful for XOR-like structure.
- `rff`: random Fourier features inspired by random kitchen sinks, useful for nonlinear boundaries without adding heavy dependencies.

Feature maps are used by the NumPy fallback backend. If TensorFlow is installed and active, Keras models ignore the feature-map selector.

Auto experiments search model settings and feature maps, then rank runs by validation F1, accuracy, and validation loss.

Experiment reports include dataset availability, class counts when a dataset is loaded, the selected threshold, fixed-`0.5` baseline metrics, calibration diagnostics, conformal-style uncertainty diagnostics, feature importance, and trial history for auto experiments.

The uncertainty output is intended as an experimental local diagnostic. When each class has enough samples, Italtensor uses a separate calibration split to estimate a split-conformal-style quantile, then evaluates coverage on the validation split. Prediction displays a label set such as `{0}`, `{1}`, `{0,1}`, or `abstain`. Tiny datasets fall back to validation-reused uncertainty and mark that source in model metadata and reports.

Batch prediction exports rank rows for review using threshold-distance uncertainty, conformal ambiguity, and row-level drift diagnostics. Rows near the decision threshold, rows whose conformal set contains both labels, or rows whose standardized features cross the OOD flag threshold get higher review priority so the file can double as a lightweight active-learning queue.

Dataset audits run from the desktop and are embedded in JSON/Markdown reports. They summarize class balance, duplicate feature rows, possible label conflicts, constant columns, highly correlated feature pairs, and compact warnings before you spend time tuning a model.

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
- Conformal-style uncertainty is strongest when the dataset is large enough for the dedicated calibration split. On tiny datasets, it falls back to validation-reused diagnostics.

## Multi-Model Controls

- **Backend** selector on the training panel chooses `auto`, `numpy`, or `keras`. `auto` uses Keras when TensorFlow is installed, otherwise NumPy.
- **Run Multi-Backend** trains one model per available backend with the same hyperparameters, stores each run in the registry, and activates the best validation F1.
- **Fusion** on the registry panel controls how ensembles combine member predictions: `mean`, `median`, `vote`, `weighted` (by slot F1), or `stacking` (linear meta-learner on validation probabilities).
- **Panel Predict** runs every stored slot on the prediction vector and logs a communication trace (per-model votes plus panel consensus and disagreement).
- **Stacked Ensemble** fits Wolpert-style stacking weights on the validation split, then adds a stacked ensemble slot.
- **Save/Load Registry** writes `italtensor.model_registry` JSON files containing NumPy models and ensembles (Keras slots must be saved individually).

## Research And Design Anchors

- Polynomial feature expansion follows the same idea exposed by scikit-learn `PolynomialFeatures`.
- Random Fourier features are based on Rahimi and Recht's random kitchen sinks work for approximating kernel methods.
- Conformal prediction design is informed by Vovk, Gammerman, and Shafer's [Algorithmic Learning in a Random World](https://link.springer.com/book/10.1007/978-3-031-06649-8), Angelopoulos and Bates' [gentle introduction](https://arxiv.org/abs/2107.07511), and Romano, Sesia, and Candes' classification-set work, [Classification with Valid and Adaptive Coverage](https://papers.nips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html).
- Batch review priority follows classic uncertainty-sampling intuition from Burr Settles' [Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf): examples closest to the model's decision boundary are often the most informative to inspect or label next.
- Batch drift flags are a lightweight standardized-distance diagnostic inspired by distance-based OOD detection work such as Lee et al.'s [simple unified framework for detecting out-of-distribution samples](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf). Italtensor uses per-row z-score summaries rather than a full covariance model to stay dependency-free.
- Multi-model stacking follows David Wolpert's stacked generalization idea: base models produce validation probabilities, and a linear meta-learner combines them ([stacked generalization](https://www.ml.cmu.edu/research/dap-papers/dap-wolpert-stacked-generalization.pdf)).
- Panel vote fusion and disagreement scores echo ensemble diversity diagnostics used in mixture-of-experts and committee-based classifiers; high disagreement flags inputs worth human review alongside conformal abstention.
- Brier/log-loss style probability evaluation and threshold tuning are standard classification-evaluation tools; Italtensor exposes both threshold-tuned metrics and fixed-threshold baselines so validation gains are visible instead of hidden.
