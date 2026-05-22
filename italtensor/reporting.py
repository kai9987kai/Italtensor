from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .modeling import ModelConfig
from .preprocessing import FeatureStandardizer


def build_experiment_report(
    *,
    sample_count: int,
    input_dim: int | None,
    labels: list[int],
    config: ModelConfig | None,
    metrics: dict[str, float | int],
    threshold: float,
    preprocessor: FeatureStandardizer | None,
    feature_importances: list[dict[str, float | int]],
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int32)
    class_counts = {
        "0": int(np.sum(label_array == 0)),
        "1": int(np.sum(label_array == 1)),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "sample_count": int(sample_count),
            "input_dim": input_dim,
            "class_counts": class_counts,
        },
        "model": {
            "config": config.to_dict() if config is not None else None,
            "threshold": float(threshold),
        },
        "preprocessing": preprocessor.to_dict() if preprocessor is not None else None,
        "metrics": metrics,
        "feature_importances": feature_importances,
    }


def export_experiment_report(path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(path)
    if output_path.suffix.lower() == ".md":
        output_path.write_text(format_markdown_report(report), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def format_markdown_report(report: dict[str, Any]) -> str:
    dataset = report.get("dataset", {})
    model = report.get("model", {})
    metrics = report.get("metrics", {})
    importances = report.get("feature_importances", [])

    lines = [
        "# Italtensor Experiment Report",
        "",
        f"Generated: {report.get('generated_at', '-')}",
        "",
        "## Dataset",
        f"- Samples: {dataset.get('sample_count', '-')}",
        f"- Input dimension: {dataset.get('input_dim', '-')}",
        f"- Class counts: {dataset.get('class_counts', {})}",
        "",
        "## Model",
        f"- Config: {model.get('config')}",
        f"- Decision threshold: {model.get('threshold', 0.5):.4f}",
        "",
        "## Metrics",
    ]
    # Core metrics (exclude cv_ prefixed and calibration for separate sections)
    core_keys = [k for k in metrics if not k.startswith("cv_")]
    calibration_keys = {"brier_score", "ece"}
    for key in core_keys:
        if key not in calibration_keys:
            lines.append(f"- {key}: {_format_value(metrics[key])}")

    # Calibration section
    if any(k in metrics for k in calibration_keys):
        lines.extend(["", "## Calibration Diagnostics"])
        for key in ["brier_score", "ece"]:
            if key in metrics:
                lines.append(f"- {key}: {_format_value(metrics[key])}")

    # Cross-validation section
    cv_keys = sorted(k for k in metrics if k.startswith("cv_"))
    if cv_keys:
        cv_folds = metrics.get("cv_folds", "?")
        lines.extend(["", f"## Cross-Validation ({cv_folds} Folds)"])
        for key in cv_keys:
            if key == "cv_folds":
                continue
            lines.append(f"- {key}: {_format_value(metrics[key])}")

    lines.extend(["", "## Top Feature Importances"])
    if importances:
        for item in importances:
            lines.append(
                f"- Feature {item.get('feature_index')}: "
                f"importance={_format_value(item.get('importance', 0.0))}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _format_value(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
