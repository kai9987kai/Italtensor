"""Export auto-experiment trial history for spreadsheets and notebooks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence


def export_trial_history_csv(path: str | Path, trial_history: Sequence[dict[str, Any]]) -> Path:
    """Write trial summaries to CSV (config fields + core validation metrics)."""
    output = Path(path)
    if not trial_history:
        raise ValueError("No trial history to export. Run Train once or auto experiments first.")

    fieldnames = [
        "trial_index",
        "backend",
        "feature_map",
        "max_epochs",
        "batch_size",
        "learning_rate",
        "mps_bond_dim",
        "threshold",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "validation_loss",
        "brier_score",
        "ece",
        "roc_auc",
        "conformal_coverage",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, trial in enumerate(trial_history, start=1):
            config = trial.get("config", {}) if isinstance(trial.get("config"), dict) else {}
            metrics = trial.get("metrics", {}) if isinstance(trial.get("metrics"), dict) else {}
            uncertainty = trial.get("uncertainty", {}) if isinstance(trial.get("uncertainty"), dict) else {}
            writer.writerow(
                {
                    "trial_index": index,
                    "backend": config.get("backend", ""),
                    "feature_map": config.get("feature_map", ""),
                    "max_epochs": config.get("max_epochs", ""),
                    "batch_size": config.get("batch_size", ""),
                    "learning_rate": config.get("learning_rate", ""),
                    "mps_bond_dim": config.get("mps_bond_dim", ""),
                    "threshold": trial.get("threshold", metrics.get("threshold", "")),
                    "f1": metrics.get("f1", ""),
                    "accuracy": metrics.get("accuracy", ""),
                    "balanced_accuracy": metrics.get("balanced_accuracy", ""),
                    "validation_loss": metrics.get("validation_loss", ""),
                    "brier_score": metrics.get("brier_score", ""),
                    "ece": metrics.get("ece", ""),
                    "roc_auc": metrics.get("roc_auc", ""),
                    "conformal_coverage": uncertainty.get("conformal_coverage", metrics.get("conformal_coverage", "")),
                }
            )
    return output
