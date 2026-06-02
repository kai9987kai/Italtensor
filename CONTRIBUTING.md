# Contributing

Italtensor is a local desktop experiment workbench for small binary-classification datasets. Keep changes practical, testable, and compatible with the NumPy fallback path unless a feature explicitly requires TensorFlow.

## Local Setup

```powershell
python -m pip install -e .[dev]
python -m pip install -r requirements.txt
```

TensorFlow-specific behavior uses `requirements-tensorflow.txt` when the full Keras backend is needed:

```powershell
python -m pip install -r requirements-tensorflow.txt
```

## Run

```powershell
python main.py
```

The GUI should stay responsive while training or diagnostics run in background workers.

## Test

Run the full suite before handing off feature work:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m pytest -q -rs
```

For narrow changes, run the focused test file first, then the full suite when the patch touches app state, reports, persistence, presets, or shared modeling behavior.

## Contribution Rules

- Avoid unsafe parsing and avoid `eval`.
- Fit preprocessing only inside training/validation splits.
- Keep diagnostics explicit about what they prove and what they do not prove.
- Preserve existing user data and model sidecar compatibility where possible.
- Add or update built-in presets when a new diagnostic needs an easy demonstration dataset.
- Add report and persistence tests whenever a feature creates evidence users may need to save.
