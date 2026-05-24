from __future__ import annotations

import csv
from pathlib import Path

import pytest

from italtensor.trials_io import export_trial_history_csv


def test_export_trial_history_csv(tmp_path: Path):
    trials = [
        {
            "config": {"backend": "numpy", "feature_map": "rff", "max_epochs": 10},
            "metrics": {"f1": 0.8, "accuracy": 0.85, "brier_score": 0.1, "ece": 0.05},
            "threshold": 0.45,
        }
    ]
    path = export_trial_history_csv(tmp_path / "trials.csv", trials)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["backend"] == "numpy"
    assert float(rows[0]["f1"]) == pytest.approx(0.8)


def test_export_trial_history_empty_raises():
    with pytest.raises(ValueError, match="No trial history"):
        export_trial_history_csv("unused.csv", [])
