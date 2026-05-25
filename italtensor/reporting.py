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
    decision_curve_report: dict[str, Any] | None = None,
    conformal_set_report: dict[str, Any] | None = None,
    calibration_repair_report: dict[str, Any] | None = None,
    selective_risk_report: dict[str, Any] | None = None,
    sample_review_report: dict[str, Any] | None = None,
    threshold_report: dict[str, Any] | None = None,
    model_response_report: dict[str, Any] | None = None,
    pairwise_interaction_report: dict[str, Any] | None = None,
    slice_report: dict[str, Any] | None = None,
    subgroup_disparity_report: dict[str, Any] | None = None,
    stress_report: dict[str, Any] | None = None,
    permutation_null_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    adversarial_validation_report: dict[str, Any] | None = None,
    cartography_report: dict[str, Any] | None = None,
    mps_sweep_report: dict[str, Any] | None = None,
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
        "decision_curve_diagnostics": decision_curve_report or None,
        "posthoc_conformal_diagnostics": conformal_set_report or None,
        "posthoc_calibration_repair_diagnostics": calibration_repair_report or None,
        "selective_prediction_diagnostics": selective_risk_report or None,
        "sample_review": sample_review_report or None,
        "threshold_diagnostics": threshold_report or None,
        "model_response_diagnostics": model_response_report or None,
        "pairwise_interaction_diagnostics": pairwise_interaction_report or None,
        "slice_diagnostics": slice_report or None,
        "subgroup_disparity_diagnostics": subgroup_disparity_report or None,
        "stress_lab": stress_report or None,
        "posthoc_permutation_null_diagnostics": permutation_null_report or None,
        "population_drift_diagnostics": population_drift_report or None,
        "adversarial_validation_diagnostics": adversarial_validation_report or None,
        "dataset_cartography": cartography_report or None,
        "mps_bond_sweep": mps_sweep_report or None,
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
    conformal_sets = report.get("posthoc_conformal_diagnostics") or {}
    calibration_repair = report.get("posthoc_calibration_repair_diagnostics") or {}
    ablation_diagnostics = report.get("feature_ablation_diagnostics") or {}
    decision_curve = report.get("decision_curve_diagnostics") or {}
    selective_risk = report.get("selective_prediction_diagnostics") or {}
    sample_review = report.get("sample_review") or {}
    threshold_diagnostics = report.get("threshold_diagnostics") or {}
    model_response = report.get("model_response_diagnostics") or {}
    pairwise_interactions = report.get("pairwise_interaction_diagnostics") or {}
    slice_diagnostics = report.get("slice_diagnostics") or {}
    subgroup_disparity = report.get("subgroup_disparity_diagnostics") or {}
    stress_lab = report.get("stress_lab") or {}
    permutation_null = report.get("posthoc_permutation_null_diagnostics") or {}
    population_drift = report.get("population_drift_diagnostics") or {}
    adversarial_validation = report.get("adversarial_validation_diagnostics") or {}
    cartography = report.get("dataset_cartography") or {}
    mps_sweep = report.get("mps_bond_sweep") or {}
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

    lines.extend(["", "## Post-Hoc Conformal Diagnostics"])
    if conformal_sets:
        summary = conformal_sets.get("summary", {})
        split = conformal_sets.get("split", {})
        lines.extend(
            [
                f"- Split source: {split.get('source', '-')}",
                f"- Calibration rows: {split.get('calibration_count', '-')}",
                f"- Evaluation rows: {split.get('evaluation_count', '-')}",
                f"- Recommended alpha: {_format_value(summary.get('recommended_alpha', '-'))}",
                f"- Target coverage: {_format_value(summary.get('recommended_target_coverage', '-'))}",
                f"- Empirical coverage: {_format_value(summary.get('recommended_empirical_coverage', '-'))}",
                f"- Mean set size: {_format_value(summary.get('recommended_mean_set_size', '-'))}",
                f"- Singleton rate: {_format_value(summary.get('recommended_singleton_rate', '-'))}",
                f"- Ambiguous rate: {_format_value(summary.get('recommended_ambiguous_rate', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in conformal_sets.get("points", [])[:8]:
            singleton_accuracy = item.get("singleton_accuracy")
            lines.append(
                f"- alpha={_format_value(item.get('alpha', '-'))}: "
                f"target={_format_value(item.get('target_coverage', '-'))}, "
                f"coverage={_format_value(item.get('empirical_coverage', '-'))}, "
                f"gap={_format_value(item.get('coverage_gap', '-'))}, "
                f"mean_size={_format_value(item.get('mean_set_size', '-'))}, "
                f"singleton_acc={_format_value(singleton_accuracy) if singleton_accuracy is not None else '-'}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Post-Hoc Calibration Repair"])
    if calibration_repair:
        summary = calibration_repair.get("summary", {})
        split = calibration_repair.get("split", {})
        lines.extend(
            [
                f"- Split source: {split.get('source', '-')}",
                f"- Calibration rows: {split.get('calibration_count', '-')}",
                f"- Evaluation rows: {split.get('evaluation_count', '-')}",
                f"- Recommended method: {summary.get('recommended_method', '-')}",
                f"- Recommended Brier: {_format_value(summary.get('recommended_brier_score', '-'))}",
                f"- Recommended ECE: {_format_value(summary.get('recommended_ece', '-'))}",
                f"- Recommended log loss: {_format_value(summary.get('recommended_log_loss', '-'))}",
                f"- Brier improvement: {_format_value(summary.get('best_brier_improvement', '-'))}",
                f"- ECE improvement: {_format_value(summary.get('best_ece_improvement', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in calibration_repair.get("methods", [])[:8]:
            lines.append(
                f"- {item.get('method', '-')}: "
                f"Brier={_format_value(item.get('brier_score', '-'))}, "
                f"ECE={_format_value(item.get('ece', '-'))}, "
                f"logloss={_format_value(item.get('log_loss', '-'))}, "
                f"dBrier={_format_value(item.get('brier_improvement', '-'))}"
            )
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

    lines.extend(["", "## Model Response / Partial Dependence"])
    if model_response:
        summary = model_response.get("summary", {})
        lines.extend(
            [
                f"- Top feature: {summary.get('top_feature', '-')}",
                f"- Top response range: {_format_value(summary.get('top_response_range', '-'))}",
                f"- Top direction: {summary.get('top_direction', '-')}",
                f"- Nonmonotonic features: {summary.get('nonmonotonic_feature_count', '-')}",
                f"- High-impact features: {summary.get('high_impact_feature_count', '-')}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in model_response.get("features", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- Feature {item.get('feature_index')}: "
                f"range={_format_value(item.get('response_range', '-'))}, "
                f"change={_format_value(item.get('signed_change', '-'))}, "
                f"direction={item.get('direction', '-')}, "
                f"min_at={_format_value(item.get('min_response_value', '-'))}, "
                f"max_at={_format_value(item.get('max_response_value', '-'))}, "
                f"flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Pairwise Feature Interactions"])
    if pairwise_interactions:
        summary = pairwise_interactions.get("summary", {})
        top_pair = summary.get("top_pair")
        pair_text = (
            f"x{int(top_pair[0]) + 1}:x{int(top_pair[1]) + 1}"
            if isinstance(top_pair, list) and len(top_pair) == 2
            else "-"
        )
        lines.extend(
            [
                f"- Evaluated pairs: {summary.get('evaluated_pair_count', '-')}",
                f"- Top pair: {pair_text}",
                f"- Top interaction strength: {_format_value(summary.get('top_interaction_strength', '-'))}",
                f"- Top max absolute interaction: {_format_value(summary.get('top_max_abs_interaction', '-'))}",
                f"- Strong pairs: {summary.get('strong_pair_count', '-')}",
                f"- Threshold-crossing pairs: {summary.get('threshold_crossing_pair_count', '-')}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in pairwise_interactions.get("pairs", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- x{int(item.get('feature_i', 0)) + 1}:x{int(item.get('feature_j', 0)) + 1}: "
                f"H={_format_value(item.get('interaction_strength', '-'))}, "
                f"max_abs={_format_value(item.get('max_abs_interaction', '-'))}, "
                f"mean_abs={_format_value(item.get('mean_abs_interaction', '-'))}, "
                f"crossings={item.get('threshold_crossings', '-')}, "
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

    lines.extend(["", "## Post-Hoc Permutation-Null Diagnostic"])
    if permutation_null:
        summary = permutation_null.get("summary", {})
        observed = permutation_null.get("observed", {})
        p_values = permutation_null.get("p_values", {})
        distribution = permutation_null.get("null_distribution", {})
        lines.extend(
            [
                f"- Permutations: {permutation_null.get('permutation_count', '-')}",
                f"- Seed: {permutation_null.get('seed', '-')}",
                f"- Verdict: {summary.get('verdict', '-')}",
                f"- Observed F1: {_format_value(summary.get('observed_f1', observed.get('f1', '-')))}",
                f"- Null mean F1: {_format_value(summary.get('null_mean_f1', '-'))}",
                f"- F1 gap: {_format_value(summary.get('f1_gap', '-'))}",
                f"- F1 z-score: {_format_value(summary.get('f1_z_score', '-'))}",
                f"- F1 p-value: {_format_value(summary.get('f1_p_value', p_values.get('f1', '-')))}",
                f"- Accuracy p-value: {_format_value(summary.get('accuracy_p_value', p_values.get('accuracy', '-')))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for metric in ("f1", "accuracy", "balanced_accuracy"):
            item = distribution.get(metric, {})
            if isinstance(item, dict):
                lines.append(
                    f"- {metric}: "
                    f"observed={_format_value(observed.get(metric, '-'))}, "
                    f"mean={_format_value(item.get('mean', '-'))}, "
                    f"p95={_format_value(item.get('p95', '-'))}, "
                    f"p={_format_value(p_values.get(metric, '-'))}"
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

    lines.extend(["", "## Decision Curve / Utility"])
    if decision_curve:
        summary = decision_curve.get("summary", {})
        current = decision_curve.get("current", {})
        ranges = summary.get("useful_threshold_ranges") or []
        range_text = ", ".join(f"{float(left):.4f}-{float(right):.4f}" for left, right in ranges) if ranges else "none"
        lines.extend(
            [
                f"- Prevalence: {_format_value(decision_curve.get('prevalence', '-'))}",
                f"- Best threshold: {_format_value(summary.get('best_threshold', '-'))}",
                f"- Best net benefit: {_format_value(summary.get('best_net_benefit', '-'))}",
                f"- Max gain vs best default: {_format_value(summary.get('max_delta_vs_best_default', '-'))}",
                f"- Useful threshold ranges: {range_text}",
                f"- Current threshold: {_format_value(current.get('threshold', '-'))}",
                f"- Current net benefit: {_format_value(current.get('net_benefit_model', '-'))}",
                f"- Current gain vs best default: {_format_value(current.get('delta_vs_best_default', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in decision_curve.get("points", [])[:8]:
            lines.append(
                f"- t={_format_value(item.get('threshold', '-'))}: "
                f"model={_format_value(item.get('net_benefit_model', '-'))}, "
                f"all={_format_value(item.get('net_benefit_treat_all', '-'))}, "
                f"none={_format_value(item.get('net_benefit_treat_none', '-'))}, "
                f"gain={_format_value(item.get('delta_vs_best_default', '-'))}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Selective Prediction / Risk-Coverage"])
    if selective_risk:
        summary = selective_risk.get("summary", {})
        base = selective_risk.get("base", {})
        lines.extend(
            [
                f"- Base risk: {_format_value(base.get('error_rate', '-'))}",
                f"- Minimum selective risk: {_format_value(summary.get('min_selective_risk', '-'))}",
                f"- Recommended cutoff: {_format_value(summary.get('recommended_cutoff', '-'))}",
                f"- Best selective accuracy: {_format_value(summary.get('best_selective_accuracy', '-'))}",
                f"- Best selective coverage: {_format_value(summary.get('best_selective_coverage', '-'))}",
                f"- Error reduction: {_format_value(summary.get('max_error_reduction', '-'))}",
                f"- Coverage at 10 pct risk: {_format_value(summary.get('coverage_at_10pct_risk', '-'))}",
                f"- AURC: {_format_value(summary.get('area_under_risk_coverage', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in selective_risk.get("ranked_cutoffs", [])[:8]:
            lines.append(
                f"- cutoff={_format_value(item.get('confidence_cutoff', '-'))}: "
                f"coverage={_format_value(item.get('coverage', '-'))}, "
                f"risk={_format_value(item.get('error_rate', '-'))}, "
                f"accuracy={_format_value(item.get('accuracy', '-'))}, "
                f"F1={_format_value(item.get('f1', '-'))}"
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

    lines.extend(["", "## Subgroup Disparity Diagnostics"])
    if subgroup_disparity:
        summary = subgroup_disparity.get("summary", {})
        lines.extend(
            [
                f"- Evaluated features: {summary.get('evaluated_feature_count', '-')}",
                f"- Evaluated subgroups: {summary.get('evaluated_subgroup_count', '-')}",
                f"- Worst feature: {summary.get('worst_feature', '-')}",
                f"- Worst subgroup: {summary.get('worst_subgroup', '-')}",
                f"- Worst metric: {summary.get('worst_metric', '-')}",
                f"- Max disparity: {_format_value(summary.get('max_disparity', '-'))}",
                f"- Max FNR gap: {_format_value(summary.get('max_false_negative_rate_gap', '-'))}",
                f"- Max FPR gap: {_format_value(summary.get('max_false_positive_rate_gap', '-'))}",
                f"- Max selection-rate gap: {_format_value(summary.get('max_predicted_positive_rate_gap', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in subgroup_disparity.get("subgroups", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- {item.get('label', '-')}: "
                f"n={item.get('count', '-')}, "
                f"coverage={_format_value(item.get('coverage', '-'))}, "
                f"gap={_format_value(item.get('risk_score', '-'))}, "
                f"metric={item.get('worst_metric', '-')}, "
                f"flags={flags}"
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

    lines.extend(["", "## Population Drift Diagnostics"])
    if population_drift:
        summary = population_drift.get("summary", {})
        label_shift = population_drift.get("label_shift", {})
        top_feature = summary.get("top_feature")
        top_text = (
            f"x{int(top_feature) + 1}"
            if isinstance(top_feature, int)
            else "-"
        )
        lines.extend(
            [
                f"- Split source: {population_drift.get('split_source', '-')}",
                f"- Reference rows: {population_drift.get('reference_count', '-')}",
                f"- Current rows: {population_drift.get('current_count', '-')}",
                f"- Top feature: {top_text}",
                f"- Max PSI: {_format_value(summary.get('max_psi', '-'))}",
                f"- Max KS statistic: {_format_value(summary.get('max_ks_statistic', '-'))}",
                f"- Max mean shift: {_format_value(summary.get('max_mean_shift_std', '-'))}",
                f"- Max outside-reference rate: {_format_value(summary.get('max_outside_reference_rate', '-'))}",
                f"- Drifted features: {summary.get('drifted_feature_count', '-')}",
                f"- Label prevalence shift: {_format_value(label_shift.get('prevalence_shift', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in population_drift.get("features", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- x{int(item.get('feature_index', 0)) + 1}: "
                f"PSI={_format_value(item.get('psi', '-'))}, "
                f"KS={_format_value(item.get('ks_statistic', '-'))}, "
                f"mean_shift={_format_value(item.get('mean_shift_std', '-'))}, "
                f"outside={_format_value(item.get('outside_reference_rate', '-'))}, "
                f"flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Adversarial Validation Diagnostics"])
    if adversarial_validation:
        summary = adversarial_validation.get("summary", {})
        metrics = adversarial_validation.get("domain_metrics", {})
        label_shift = adversarial_validation.get("label_shift", {})
        top_feature = summary.get("top_feature")
        top_text = (
            f"x{int(top_feature) + 1}"
            if isinstance(top_feature, int)
            else "-"
        )
        lines.extend(
            [
                f"- Split source: {adversarial_validation.get('split_source', '-')}",
                f"- Reference rows: {adversarial_validation.get('reference_count', '-')}",
                f"- Current rows: {adversarial_validation.get('current_count', '-')}",
                f"- Validation rows: {adversarial_validation.get('validation_samples', '-')}",
                f"- Domain AUC: {_format_value(summary.get('domain_auc', metrics.get('roc_auc', '-')))}",
                f"- Domain accuracy: {_format_value(summary.get('domain_accuracy', metrics.get('accuracy', '-')))}",
                f"- Detectability: {_format_value(summary.get('detectability', '-'))}",
                f"- Verdict: {summary.get('verdict', '-')}",
                f"- Top feature: {top_text}",
                f"- Important features: {summary.get('important_feature_count', '-')}",
                f"- Label prevalence shift: {_format_value(label_shift.get('prevalence_shift', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in adversarial_validation.get("features", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- x{int(item.get('feature_index', 0)) + 1}: "
                f"auc_drop={_format_value(item.get('auc_drop', '-'))}, "
                f"accuracy_drop={_format_value(item.get('accuracy_drop', '-'))}, "
                f"prob_shift={_format_value(item.get('mean_probability_shift', '-'))}, "
                f"flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Dataset Cartography"])
    if cartography:
        counts = cartography.get("region_counts", {})
        lines.extend(
            [
                f"- Samples: {cartography.get('sample_count', '-')}",
                f"- Threshold: {_format_value(cartography.get('threshold', '-'))}",
                f"- Median confidence: {_format_value(cartography.get('median_confidence', '-'))}",
                f"- Median variability: {_format_value(cartography.get('median_variability', '-'))}",
                f"- Easy rows: {counts.get('easy_to_learn', '-')}",
                f"- Ambiguous rows: {counts.get('ambiguous', '-')}",
                f"- Hard rows: {counts.get('hard_to_learn', '-')}",
                f"- Overconfident wrong rows: {counts.get('overconfident_wrong', '-')}",
            ]
        )
        for region_name in ("overconfident_wrong", "ambiguous", "hard_to_learn", "easy_to_learn"):
            for item in cartography.get("regions", {}).get(region_name, [])[:3]:
                lines.append(
                    f"- {region_name} row {item.get('row_index')}: "
                    f"label={item.get('label')}, pred={item.get('predicted_label')}, "
                    f"conf={_format_value(item.get('confidence', '-'))}, "
                    f"var={_format_value(item.get('variability', '-'))}"
                )
    else:
        lines.append("- None")

    lines.extend(["", "## MPS Bond Sweep"])
    if mps_sweep:
        lines.extend(
            [
                f"- Input dimension: {mps_sweep.get('input_dim', '-')}",
                f"- Physical dimension: {mps_sweep.get('physical_dim', '-')}",
                f"- Validation samples: {mps_sweep.get('validation_samples', '-')}",
                f"- Tested chi: {mps_sweep.get('bond_dims_tested', [])}",
                f"- Recommended chi: {mps_sweep.get('recommended_bond_dim', '-')}",
                f"- Recommended F1: {_format_value(mps_sweep.get('recommended_f1', '-'))}",
            ]
        )
        for item in mps_sweep.get("results", [])[:8]:
            lines.append(
                f"- chi={item.get('bond_dim', '-')}: "
                f"F1={_format_value(item.get('f1', '-'))}, "
                f"accuracy={_format_value(item.get('accuracy', '-'))}, "
                f"Brier={_format_value(item.get('brier_score', '-'))}, "
                f"ECE={_format_value(item.get('ece', '-'))}"
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
