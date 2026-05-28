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
    error_atlas_report: dict[str, Any] | None = None,
    threshold_report: dict[str, Any] | None = None,
    model_response_report: dict[str, Any] | None = None,
    pairwise_interaction_report: dict[str, Any] | None = None,
    slice_report: dict[str, Any] | None = None,
    subgroup_disparity_report: dict[str, Any] | None = None,
    stress_report: dict[str, Any] | None = None,
    permutation_null_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    adversarial_validation_report: dict[str, Any] | None = None,
    chronological_holdout_report: dict[str, Any] | None = None,
    cartography_report: dict[str, Any] | None = None,
    ood_sentinel_report: dict[str, Any] | None = None,
    bootstrap_stability_report: dict[str, Any] | None = None,
    prototype_audit_report: dict[str, Any] | None = None,
    feature_separability_report: dict[str, Any] | None = None,
    neighborhood_hardness_report: dict[str, Any] | None = None,
    dataset_triage_report: dict[str, Any] | None = None,
    experiment_advisor_report: dict[str, Any] | None = None,
    trial_inspector_report: dict[str, Any] | None = None,
    promotion_gate_report: dict[str, Any] | None = None,
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
        "error_atlas": error_atlas_report or None,
        "threshold_diagnostics": threshold_report or None,
        "model_response_diagnostics": model_response_report or None,
        "pairwise_interaction_diagnostics": pairwise_interaction_report or None,
        "slice_diagnostics": slice_report or None,
        "subgroup_disparity_diagnostics": subgroup_disparity_report or None,
        "stress_lab": stress_report or None,
        "posthoc_permutation_null_diagnostics": permutation_null_report or None,
        "population_drift_diagnostics": population_drift_report or None,
        "adversarial_validation_diagnostics": adversarial_validation_report or None,
        "chronological_holdout_diagnostics": chronological_holdout_report or None,
        "dataset_cartography": cartography_report or None,
        "ood_sentinel": ood_sentinel_report or None,
        "bootstrap_stability_diagnostics": bootstrap_stability_report or None,
        "prototype_audit": prototype_audit_report or None,
        "feature_separability": feature_separability_report or None,
        "neighborhood_hardness": neighborhood_hardness_report or None,
        "dataset_triage": dataset_triage_report or None,
        "experiment_advisor": experiment_advisor_report or None,
        "trial_inspector": trial_inspector_report or None,
        "promotion_gate": promotion_gate_report or None,
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
    error_atlas = report.get("error_atlas") or {}
    threshold_diagnostics = report.get("threshold_diagnostics") or {}
    model_response = report.get("model_response_diagnostics") or {}
    pairwise_interactions = report.get("pairwise_interaction_diagnostics") or {}
    slice_diagnostics = report.get("slice_diagnostics") or {}
    subgroup_disparity = report.get("subgroup_disparity_diagnostics") or {}
    stress_lab = report.get("stress_lab") or {}
    permutation_null = report.get("posthoc_permutation_null_diagnostics") or {}
    population_drift = report.get("population_drift_diagnostics") or {}
    adversarial_validation = report.get("adversarial_validation_diagnostics") or {}
    chronological_holdout = report.get("chronological_holdout_diagnostics") or {}
    cartography = report.get("dataset_cartography") or {}
    ood_sentinel = report.get("ood_sentinel") or {}
    bootstrap_stability = report.get("bootstrap_stability_diagnostics") or {}
    prototype_audit = report.get("prototype_audit") or {}
    feature_separability = report.get("feature_separability") or {}
    neighborhood_hardness = report.get("neighborhood_hardness") or {}
    dataset_triage = report.get("dataset_triage") or {}
    experiment_advisor = report.get("experiment_advisor") or {}
    trial_inspector = report.get("trial_inspector") or {}
    promotion_gate = report.get("promotion_gate") or {}
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

    lines.extend(["", "## Dataset Triage"])
    if dataset_triage:
        summary = dataset_triage.get("summary", {})
        lines.extend(
            [
                f"- Readiness score: {_format_value(summary.get('readiness_score', '-'))}/100",
                f"- Risk level: {summary.get('risk_level', '-')}",
                f"- Blocking issues: {summary.get('blocking_issue_count', '-')}",
                f"- Penalty points: {_format_value(summary.get('penalty_points', '-'))}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for action in summary.get("top_actions", [])[:6]:
            lines.append(f"- Action: {action}")
    else:
        lines.append("- None")

    lines.extend(["", "## Experiment Advisor"])
    if experiment_advisor:
        summary = experiment_advisor.get("summary", {})
        lines.extend(
            [
                f"- Recommendations: {summary.get('recommendation_count', '-')}",
                f"- Top priority: {summary.get('top_priority', '-')}",
                f"- Top category: {summary.get('top_category', '-')}",
                f"- Recommended next step: {summary.get('recommended_next_step') or 'none'}",
                f"- Needs training: {summary.get('needs_training', '-')}",
            ]
        )
        for item in experiment_advisor.get("recommendations", [])[:8]:
            lines.append(
                f"- {item.get('rank', '-')}. [{item.get('priority', '-')}/{item.get('category', '-')}] "
                f"{item.get('title', '-')}: {item.get('action', '-')}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Trial Inspector"])
    if trial_inspector:
        summary = trial_inspector.get("summary", {})
        lines.extend(
            [
                f"- Valid trials: {trial_inspector.get('valid_trial_count', '-')}/{trial_inspector.get('trial_count', '-')}",
                f"- Best trial: {summary.get('best_trial_index', '-')}",
                f"- Best backend/map: {summary.get('best_backend', '-')}/{summary.get('best_feature_map', '-')}",
                f"- Best F1: {_format_value(summary.get('best_f1', '-'))}",
                f"- Leader margin F1: {_format_value(summary.get('leader_margin_f1', '-'))}",
                f"- Recommendation: {summary.get('recommendation') or 'none'}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in trial_inspector.get("leaderboard", [])[:5]:
            lines.append(
                f"- Rank {item.get('rank', '-')}: trial {item.get('trial_index', '-')} "
                f"{item.get('backend', '-')}/{item.get('feature_map', '-')} "
                f"F1={_format_value(item.get('f1', '-'))} "
                f"accuracy={_format_value(item.get('accuracy', '-'))} "
                f"loss={_format_value(item.get('validation_loss', '-'))}"
            )
        for group in trial_inspector.get("groups", [])[:4]:
            lines.append(
                f"- Group {group.get('group', '-')}: count={group.get('count', '-')} "
                f"best_F1={_format_value(group.get('best_f1', '-'))} "
                f"avg_F1={_format_value(group.get('avg_f1', '-'))}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Promotion Gate"])
    if promotion_gate:
        summary = promotion_gate.get("summary", {})
        lines.extend(
            [
                f"- Verdict: {summary.get('verdict', '-')}",
                f"- Promotion score: {_format_value(summary.get('promotion_score', '-'))}/100",
                f"- Blockers: {summary.get('blocker_count', '-')}",
                f"- Cautions: {summary.get('caution_count', '-')}",
                f"- Required next step: {summary.get('required_next_step') or 'none'}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in promotion_gate.get("checks", [])[:8]:
            lines.append(
                f"- {item.get('rank', '-')}. [{item.get('severity', '-')}/{item.get('category', '-')}] "
                f"{item.get('title', '-')}: {item.get('action', '-')}"
            )
        release_note = promotion_gate.get("release_note", {})
        if release_note:
            lines.append(f"- Recommended use: {release_note.get('recommended_use', '-')}")
            lines.append(f"- Must include: {release_note.get('must_include', [])}")
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

    lines.extend(["", "## Error Atlas"])
    if error_atlas:
        summary = error_atlas.get("summary", {})
        confusion = error_atlas.get("confusion", {})
        lines.extend(
            [
                f"- Errors: {summary.get('error_count', '-')}/{error_atlas.get('sample_count', '-')}",
                f"- Error rate: {_format_value(summary.get('error_rate', '-'))}",
                f"- False positives: {confusion.get('false_positive', '-')}",
                f"- False negatives: {confusion.get('false_negative', '-')}",
                f"- High-confidence errors: {summary.get('high_confidence_error_count', '-')}",
                f"- Near-threshold rows: {summary.get('near_threshold_count', '-')}",
                f"- Dominant error type: {summary.get('dominant_error_type', '-')}",
                f"- Recommendation: {summary.get('recommendation') or 'none'}",
            ]
        )
        for label, rows in (
            ("high-confidence error", error_atlas.get("high_confidence_errors", [])),
            ("near-threshold", error_atlas.get("near_threshold_rows", [])),
        ):
            for item in rows[:5]:
                lines.append(
                    f"- {label} row {item.get('row_index')}: "
                    f"label={item.get('label')}, pred={item.get('predicted_label')}, "
                    f"p={_format_value(item.get('probability', '-'))}, "
                    f"loss={_format_value(item.get('loss', '-'))}, "
                    f"margin={_format_value(item.get('margin', '-'))}"
                )
        for item in error_atlas.get("feature_error_shifts", [])[:5]:
            lines.append(
                f"- Error-shift x{int(item.get('feature_index', 0)) + 1}: "
                f"shift={_format_value(item.get('standardized_shift', '-'))}, "
                f"error_mean={_format_value(item.get('error_mean', '-'))}, "
                f"correct_mean={_format_value(item.get('correct_mean', '-'))}"
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

    lines.extend(["", "## Chronological Holdout Diagnostics"])
    if chronological_holdout:
        summary = chronological_holdout.get("summary", {})
        reference_metrics = chronological_holdout.get("reference_metrics", {})
        current_metrics = chronological_holdout.get("current_metrics", {})
        metric_deltas = chronological_holdout.get("metric_deltas", {})
        probability_diagnostics = chronological_holdout.get("current_probability_diagnostics", {})
        label_shift = chronological_holdout.get("label_shift", {})
        baseline = chronological_holdout.get("current_baseline", {})
        top_feature = summary.get("top_current_reliance_feature")
        top_text = (
            f"x{int(top_feature) + 1}"
            if isinstance(top_feature, int)
            else "-"
        )
        lines.extend(
            [
                f"- Split source: {chronological_holdout.get('split_source', '-')}",
                f"- Reference rows: {chronological_holdout.get('reference_count', '-')}",
                f"- Reference evaluation rows: {chronological_holdout.get('reference_evaluation_count', '-')}",
                f"- Current rows: {chronological_holdout.get('current_count', '-')}",
                f"- Feature map: {chronological_holdout.get('feature_map', '-')}",
                f"- Threshold: {_format_value(chronological_holdout.get('threshold', '-'))}",
                f"- Reference F1: {_format_value(reference_metrics.get('f1', '-'))}",
                f"- Current F1: {_format_value(current_metrics.get('f1', '-'))}",
                f"- F1 delta: {_format_value(metric_deltas.get('f1_delta', '-'))}",
                f"- Accuracy delta: {_format_value(metric_deltas.get('accuracy_delta', '-'))}",
                f"- Brier delta: {_format_value(metric_deltas.get('brier_score_delta', '-'))}",
                f"- Log loss delta: {_format_value(metric_deltas.get('log_loss_delta', '-'))}",
                f"- Mean probability delta: {_format_value(probability_diagnostics.get('mean_probability_delta', '-'))}",
                f"- Current ECE: {_format_value(probability_diagnostics.get('current_ece', '-'))}",
                f"- Label prevalence shift: {_format_value(label_shift.get('prevalence_shift', '-'))}",
                f"- Top current reliance feature: {top_text}",
                f"- Current-baseline F1 gain: {_format_value(summary.get('current_baseline_f1_gain', '-'))}",
                f"- Verdict: {summary.get('verdict', '-')}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        if baseline.get("available"):
            baseline_metrics = baseline.get("current_model_metrics", {})
            baseline_deltas = baseline.get("metric_deltas_vs_reference_model", {})
            lines.extend(
                [
                    f"- Current-baseline train rows: {baseline.get('current_train_count', '-')}",
                    f"- Current-baseline evaluation rows: {baseline.get('current_evaluation_count', '-')}",
                    f"- Current-baseline F1: {_format_value(baseline_metrics.get('f1', '-'))}",
                    f"- Current-baseline F1 gain vs reference model: {_format_value(baseline_deltas.get('f1_delta', '-'))}",
                ]
            )
        else:
            lines.append(f"- Current-baseline unavailable: {baseline.get('reason', '-')}")
        for item in chronological_holdout.get("permutation_reliance", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- x{int(item.get('feature_index', 0)) + 1}: "
                f"F1_drop={_format_value(item.get('f1_drop', '-'))}, "
                f"logloss_increase={_format_value(item.get('log_loss_increase', '-'))}, "
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

    lines.extend(["", "## Neighborhood Hardness"])
    if neighborhood_hardness:
        summary = neighborhood_hardness.get("summary", {})
        top_row = summary.get("top_hard_row")
        lines.extend(
            [
                f"- Rows scanned: {neighborhood_hardness.get('sample_count', '-')}",
                f"- k: {neighborhood_hardness.get('k', '-')}",
                f"- Leave-one-out accuracy: {_format_value(summary.get('loo_accuracy', '-'))}",
                f"- Hard rows: {summary.get('hard_row_count', '-')}",
                f"- Ambiguous rows: {summary.get('ambiguous_row_count', '-')}",
                f"- Label issue candidates: {summary.get('label_issue_candidate_count', '-')}",
                f"- Locally easy rows: {summary.get('locally_easy_count', '-')}",
                f"- Top hard row: {top_row if top_row is not None else '-'}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in neighborhood_hardness.get("rows", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- row {item.get('row_index', '-')}: "
                f"label={item.get('label', '-')}, "
                f"vote={item.get('predicted_label', '-')}, "
                f"hardness={_format_value(item.get('hardness_score', '-'))}, "
                f"opp_vote={_format_value(item.get('opposite_vote_rate', '-'))}, "
                f"entropy={_format_value(item.get('vote_entropy', '-'))}, "
                f"flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Feature Separability Lens"])
    if feature_separability:
        summary = feature_separability.get("summary", {})
        top_feature = summary.get("top_feature")
        top_text = "-" if top_feature is None else f"x{int(top_feature) + 1}"
        lines.extend(
            [
                f"- Rows scanned: {feature_separability.get('sample_count', '-')}",
                f"- Input dimension: {feature_separability.get('input_dim', '-')}",
                f"- Top feature: {top_text}",
                f"- Top AUC: {_format_value(summary.get('top_auc', '-'))}",
                f"- Top balanced accuracy: {_format_value(summary.get('top_balanced_accuracy', '-'))}",
                f"- Near-perfect features: {summary.get('near_perfect_feature_count', '-')}",
                f"- Weak features: {summary.get('weak_feature_count', '-')}",
                f"- Redundant pairs: {summary.get('redundant_pair_count', '-')}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in feature_separability.get("features", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- x{int(item.get('feature_index', 0)) + 1}: "
                f"AUC={_format_value(item.get('auc', '-'))}, "
                f"bal_acc={_format_value(item.get('best_balanced_accuracy', '-'))}, "
                f"SMD={_format_value(item.get('standardized_mean_difference', '-'))}, "
                f"direction={item.get('direction', '-')}, "
                f"flags={flags}"
            )
        for item in feature_separability.get("redundant_pairs", [])[:5]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- redundant x{int(item.get('left_feature_index', 0)) + 1}/x{int(item.get('right_feature_index', 0)) + 1}: "
                f"corr={_format_value(item.get('correlation', '-'))}, flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Prototype Audit"])
    if prototype_audit:
        summary = prototype_audit.get("summary", {})
        lines.extend(
            [
                f"- Rows scanned: {prototype_audit.get('sample_count', '-')}",
                f"- k: {prototype_audit.get('k', '-')}",
                f"- Prototypes: {summary.get('prototype_count', '-')}",
                f"- Boundary rows: {summary.get('boundary_row_count', '-')}",
                f"- Isolated rows: {summary.get('isolated_row_count', '-')}",
                f"- Possible label contradictions: {summary.get('label_contradiction_count', '-')}",
                f"- Top boundary row: {summary.get('top_boundary_row', '-')}",
                f"- Top contradiction row: {summary.get('top_label_contradiction_row', '-')}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in prototype_audit.get("prototypes", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- prototype row {item.get('row_index', '-')}: "
                f"label={item.get('label', '-')}, "
                f"score={_format_value(item.get('prototype_score', '-'))}, "
                f"opp_frac={_format_value(item.get('local_opposite_fraction', '-'))}, "
                f"flags={flags}"
            )
        for item in prototype_audit.get("boundary_rows", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- boundary row {item.get('row_index', '-')}: "
                f"label={item.get('label', '-')}, "
                f"boundary={_format_value(item.get('boundary_score', '-'))}, "
                f"contradiction={_format_value(item.get('label_contradiction_score', '-'))}, "
                f"flags={flags}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## OOD Sentinel"])
    if ood_sentinel:
        summary = ood_sentinel.get("summary", {})
        top_row = summary.get("top_row_index")
        lines.extend(
            [
                f"- Model used: {ood_sentinel.get('model_used', False)}",
                f"- Rows scanned: {ood_sentinel.get('sample_count', '-')}",
                f"- Top row: {top_row if top_row is not None else '-'}",
                f"- Max OOD score: {_format_value(summary.get('max_ood_score', '-'))}",
                f"- Max robust z: {_format_value(summary.get('max_abs_robust_z', '-'))}",
                f"- Max nearest-neighbor distance: {_format_value(summary.get('max_nearest_neighbor_distance', '-'))}",
                f"- Flagged rows: {summary.get('flagged_row_count', '-')}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in ood_sentinel.get("rows", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            details = (
                f"- row {item.get('row_index', '-')}: "
                f"score={_format_value(item.get('ood_score', '-'))}, "
                f"max_z={_format_value(item.get('max_abs_robust_z', '-'))}, "
                f"nn={_format_value(item.get('nearest_neighbor_distance', '-'))}"
            )
            if item.get("loss") is not None:
                details += (
                    f", loss={_format_value(item.get('loss', '-'))}, "
                    f"p={_format_value(item.get('probability', '-'))}"
                )
            lines.append(f"{details}, flags={flags}")
    else:
        lines.append("- None")

    lines.extend(["", "## Bootstrap Stability Diagnostics"])
    if bootstrap_stability:
        summary = bootstrap_stability.get("summary", {})
        metrics = bootstrap_stability.get("ensemble_metrics", {})
        top_row = summary.get("top_row_index")
        lines.extend(
            [
                f"- Models: {bootstrap_stability.get('model_count', '-')}",
                f"- Feature map: {bootstrap_stability.get('feature_map', '-')}",
                f"- Threshold: {_format_value(bootstrap_stability.get('threshold', '-'))}",
                f"- Ensemble F1: {_format_value(metrics.get('f1', '-'))}",
                f"- Ensemble accuracy: {_format_value(metrics.get('accuracy', '-'))}",
                f"- Mean probability std: {_format_value(summary.get('mean_probability_std', '-'))}",
                f"- Max probability std: {_format_value(summary.get('max_probability_std', '-'))}",
                f"- Max disagreement: {_format_value(summary.get('max_disagreement_rate', '-'))}",
                f"- Unstable rows: {summary.get('unstable_row_count', '-')}",
                f"- Top row: {top_row if top_row is not None else '-'}",
                f"- Warning: {summary.get('warning') or 'none'}",
            ]
        )
        for item in bootstrap_stability.get("rows", [])[:8]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            lines.append(
                f"- row {item.get('row_index', '-')}: "
                f"instability={_format_value(item.get('instability_score', '-'))}, "
                f"std={_format_value(item.get('probability_std', '-'))}, "
                f"disagreement={_format_value(item.get('disagreement_rate', '-'))}, "
                f"mean_p={_format_value(item.get('mean_probability', '-'))}, "
                f"flags={flags}"
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
