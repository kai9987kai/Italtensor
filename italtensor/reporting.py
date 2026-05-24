from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .audit import audit_dataset
from .modeling import ModelConfig
from .preprocessing import FeatureStandardizer


def build_experiment_report(
    *,
    sample_count: int,
    input_dim: int | None,
    labels: list[int],
    features: list[list[float]] | None = None,
    config: ModelConfig | None,
    metrics: dict[str, float | int],
    threshold: float,
    preprocessor: FeatureStandardizer | None,
    feature_importances: list[dict[str, float | int]],
    trial_history: list[dict[str, Any]] | None = None,
    uncertainty_metadata: dict[str, Any] | None = None,
    ablation_report: dict[str, Any] | None = None,
    sample_review_report: dict[str, Any] | None = None,
    threshold_report: dict[str, Any] | None = None,
    slice_report: dict[str, Any] | None = None,
    stress_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int32)
    dataset_available = bool(sample_count and label_array.size)
    class_counts = (
        {
            "0": int(np.sum(label_array == 0)),
            "1": int(np.sum(label_array == 1)),
        }
        if dataset_available
        else None
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "available": dataset_available,
            "sample_count": int(sample_count),
            "input_dim": input_dim,
            "class_counts": class_counts,
            "audit": audit_dataset(features, labels) if features is not None and labels else None,
        },
        "model": {
            "config": config.to_dict() if config is not None else None,
            "threshold": float(threshold),
        },
        "preprocessing": preprocessor.to_dict() if preprocessor is not None else None,
        "metrics": metrics,
        "uncertainty": uncertainty_metadata or None,
        "feature_ablation_diagnostics": ablation_report or None,
        "sample_review": sample_review_report or None,
        "threshold_diagnostics": threshold_report or None,
        "slice_diagnostics": slice_report or None,
        "stress_lab": stress_report or None,
        "feature_importances": feature_importances,
        "trial_history": trial_history or [],
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
    trial_history = report.get("trial_history", [])
    uncertainty = report.get("uncertainty") or {}
    ablation_diagnostics = report.get("feature_ablation_diagnostics") or {}
    sample_review = report.get("sample_review") or {}
    threshold_diagnostics = report.get("threshold_diagnostics") or {}
    slice_diagnostics = report.get("slice_diagnostics") or {}
    stress_lab = report.get("stress_lab") or {}
    audit = dataset.get("audit") or {}

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
        "## Dataset Audit",
    ]
    if audit:
        lines.extend(
            [
                f"- Imbalance ratio: {_format_value(audit.get('imbalance_ratio', '-'))}",
                f"- Duplicate rows: {audit.get('duplicate_row_count', '-')}",
                f"- Duplicate groups: {audit.get('duplicate_rows', {}).get('duplicate_group_count', '-')}",
                f"- Label conflicts: {audit.get('label_conflict_count', '-')}",
                f"- Conflicting rows: {audit.get('conflicting_row_count', '-')}",
                f"- Constant features: {audit.get('constant_features', [])}",
                f"- High correlation pairs: {len(audit.get('high_correlations', []))}",
                f"- Warnings: {audit.get('warnings', [])}",
            ]
        )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Model",
        f"- Config: {model.get('config')}",
        f"- Decision threshold: {model.get('threshold', 0.5):.4f}",
        "",
        "## Metrics",
    ])
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

    lines.extend(["", "## Uncertainty"])
    if uncertainty:
        for key in [
            "conformal_source",
            "conformal_alpha",
            "conformal_quantile",
            "conformal_target_coverage",
            "conformal_coverage",
            "conformal_calibration_count",
            "conformal_evaluation_count",
            "conformal_singleton_rate",
            "conformal_empty_rate",
            "conformal_both_rate",
            "conformal_mean_set_size",
        ]:
            if key in uncertainty:
                lines.append(f"- {key}: {_format_value(uncertainty[key])}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Feature Importances"])
    if importances:
        for item in importances:
            lines.append(
                f"- Feature {item.get('feature_index')}: "
                f"importance={_format_value(item.get('importance', 0.0))}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Ablation Diagnostics"])
    if ablation_diagnostics:
        summary = ablation_diagnostics.get("summary", {})
        base = ablation_diagnostics.get("base", {})
        lines.extend(
            [
                f"- Base F1: {_format_value(base.get('f1', '-'))}",
                f"- Top feature: {summary.get('top_feature', '-')}",
                f"- Max F1 drop: {_format_value(summary.get('max_f1_drop', '-'))}",
                f"- Max label flip rate: {_format_value(summary.get('max_label_flip_rate', '-'))}",
                f"- High-reliance features: {summary.get('high_reliance_count', '-')}",
                f"- Label-proxy flags: {summary.get('label_proxy_count', '-')}",
            ]
        )
        for item in ablation_diagnostics.get("features", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- Feature {item.get('feature_index')}: "
                f"drop={_format_value(item.get('f1_drop', '-'))}, "
                f"perm_drop={_format_value(item.get('permutation_f1_drop', '-'))}, "
                f"flip={_format_value(max(float(item.get('label_flip_rate', 0.0)), float(item.get('permutation_label_flip_rate', 0.0))))}, "
                f"corr={_format_value(item.get('label_correlation', '-'))}, "
                f"flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Sample Review"])
    if sample_review:
        summary = sample_review.get("summary", {})
        lines.extend(
            [
                f"- Label issues: {summary.get('label_issue_count', '-')}",
                f"- Disagreements: {summary.get('disagreement_count', '-')}",
                f"- Ambiguous rows: {summary.get('ambiguous_count', '-')}",
                f"- Mean loss: {_format_value(summary.get('mean_loss', '-'))}",
                f"- Max loss: {_format_value(summary.get('max_loss', '-'))}",
            ]
        )
        for label, key in (("label_issue", "label_issues"), ("hard", "hard_examples"), ("ambiguous", "ambiguous_examples")):
            for item in sample_review.get(key, [])[:5]:
                lines.append(
                    f"- {label} row {item.get('row_index')}: "
                    f"label={item.get('label')}, pred={item.get('predicted_label')}, "
                    f"p={_format_value(item.get('probability', '-'))}, "
                    f"loss={_format_value(item.get('loss', '-'))}"
                )
    else:
        lines.append("- None")

    lines.extend(["", "## Threshold Tradeoffs"])
    if threshold_diagnostics:
        summary = threshold_diagnostics.get("summary", {})
        lines.extend(
            [
                f"- Current threshold: {_format_value(threshold_diagnostics.get('current_threshold', '-'))}",
                f"- Best F1 threshold: {_format_value(summary.get('best_f1_threshold', '-'))}",
                f"- Best balanced-accuracy threshold: {_format_value(summary.get('best_balanced_accuracy_threshold', '-'))}",
                f"- Minimum-cost threshold: {_format_value(summary.get('min_cost_threshold', '-'))}",
                f"- Current cost: {_format_value(summary.get('current_cost', '-'))}",
                f"- Minimum cost: {_format_value(summary.get('min_cost', '-'))}",
            ]
        )
        for label in ("best_f1", "best_balanced_accuracy", "min_cost", "high_recall", "high_precision"):
            item = threshold_diagnostics.get(label)
            if isinstance(item, dict):
                lines.append(
                    f"- {label}: t={_format_value(item.get('threshold', '-'))}, "
                    f"F1={_format_value(item.get('f1', '-'))}, "
                    f"precision={_format_value(item.get('precision', '-'))}, "
                    f"recall={_format_value(item.get('recall', '-'))}, "
                    f"cost={_format_value(item.get('cost', '-'))}"
                )
    else:
        lines.append("- None")

    lines.extend(["", "## Slice Diagnostics"])
    if slice_diagnostics:
        summary = slice_diagnostics.get("summary", {})
        base = slice_diagnostics.get("base", {})
        lines.extend(
            [
                f"- Base F1: {_format_value(base.get('f1', '-'))}",
                f"- Slice count: {summary.get('slice_count', '-')}",
                f"- Worst slice: {summary.get('worst_slice', '-')}",
                f"- Worst F1 delta: {_format_value(summary.get('worst_f1_delta', '-'))}",
                f"- Worst accuracy delta: {_format_value(summary.get('worst_accuracy_delta', '-'))}",
            ]
        )
        for item in slice_diagnostics.get("slices", [])[:8]:
            lines.append(
                f"- x{int(item.get('feature_index', 0)) + 1}"
                f"[{_format_value(item.get('left', '-'))}, {_format_value(item.get('right', '-'))}]: "
                f"n={item.get('count', '-')}, "
                f"F1={_format_value(item.get('f1', '-'))}, "
                f"delta={_format_value(item.get('f1_delta', '-'))}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Robustness Stress Lab"])
    if stress_lab:
        summary = stress_lab.get("summary", {})
        base = stress_lab.get("base", {})
        lines.extend(
            [
                f"- Base F1: {_format_value(base.get('f1', '-'))}",
                f"- Worst F1: {_format_value(summary.get('worst_f1', '-'))}",
                f"- Stress F1 ratio: {_format_value(summary.get('stress_f1_ratio', '-'))}",
                f"- Max label flip rate: {_format_value(summary.get('max_label_flip_rate', '-'))}",
                f"- Worst case: {summary.get('worst_case', '-')}",
            ]
        )
        for item in stress_lab.get("perturbations", [])[:8]:
            label = item.get("kind", "-")
            if "feature_index" in item:
                label = f"{label}[x{int(item['feature_index']) + 1}]"
            lines.append(
                f"- {label}@{_format_value(item.get('level', '-'))}: "
                f"F1={_format_value(item.get('f1', '-'))}, "
                f"flip={_format_value(item.get('label_flip_rate', '-'))}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Trial History"])
    if trial_history:
        for index, trial in enumerate(trial_history, start=1):
            config = trial.get("config", {})
            trial_metrics = trial.get("metrics", {})
            lines.append(
                f"- Trial {index}: map={config.get('feature_map', '-')}, "
                f"f1={_format_value(trial_metrics.get('f1', '-'))}, "
                f"brier={_format_value(trial_metrics.get('brier_score', '-'))}, "
                f"log_loss={_format_value(trial_metrics.get('log_loss', trial_metrics.get('validation_loss', '-')))}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _format_value(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)
