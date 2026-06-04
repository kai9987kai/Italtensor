from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import ModelConfig


def build_experiment_advisor(
    *,
    sample_count: int,
    input_dim: int | None,
    labels: Sequence[int] | np.ndarray,
    config: ModelConfig | None = None,
    metrics: dict[str, float | int] | None = None,
    trial_history: list[dict[str, Any]] | None = None,
    dataset_triage_report: dict[str, Any] | None = None,
    leakage_sentinel_report: dict[str, Any] | None = None,
    feature_separability_report: dict[str, Any] | None = None,
    neighborhood_hardness_report: dict[str, Any] | None = None,
    prototype_audit_report: dict[str, Any] | None = None,
    ood_sentinel_report: dict[str, Any] | None = None,
    threshold_report: dict[str, Any] | None = None,
    calibration_repair_report: dict[str, Any] | None = None,
    calibration_slice_report: dict[str, Any] | None = None,
    external_holdout_report: dict[str, Any] | None = None,
    decision_curve_report: dict[str, Any] | None = None,
    selective_risk_report: dict[str, Any] | None = None,
    stress_report: dict[str, Any] | None = None,
    permutation_null_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    adversarial_validation_report: dict[str, Any] | None = None,
    chronological_holdout_report: dict[str, Any] | None = None,
    learning_curve_report: dict[str, Any] | None = None,
    data_acquisition_report: dict[str, Any] | None = None,
    data_value_report: dict[str, Any] | None = None,
    max_recommendations: int = 8,
) -> dict[str, Any]:
    """Rank practical next experiments from current dataset, model, and diagnostic evidence."""
    max_recommendations = max(1, int(max_recommendations))
    metrics = metrics or {}
    trial_history = trial_history or []
    labels_array = _labels_to_array(labels)
    class_counts = _class_counts(labels_array) if labels_array.size else None
    base_config = (config or ModelConfig()).to_dict()
    recommendations: list[dict[str, Any]] = []

    def add(
        *,
        score: float,
        priority: str,
        category: str,
        title: str,
        reason: str,
        action: str,
        source: str,
        suggested_config: dict[str, Any] | None = None,
        expected_signal: str | None = None,
    ) -> None:
        recommendations.append(
            {
                "priority": priority,
                "priority_score": float(score),
                "category": category,
                "title": title,
                "reason": reason,
                "action": action,
                "source": source,
                "suggested_config": suggested_config or None,
                "expected_signal": expected_signal,
            }
        )

    if sample_count <= 0 or labels_array.size == 0:
        add(
            score=100.0,
            priority="high",
            category="data",
            title="Load or create a labeled dataset",
            reason="No dataset is available, so training, diagnostics, and reports have no evidence base.",
            action="Load a preset, import a CSV, or add JSON samples before running experiments.",
            source="dataset",
        )
        return _finalize(sample_count, input_dim, class_counts, metrics, trial_history, recommendations, max_recommendations)

    triage_summary = (dataset_triage_report or {}).get("summary", {})
    for index, action in enumerate(triage_summary.get("top_actions", [])[:3]):
        risk = str(triage_summary.get("risk_level", "medium"))
        add(
            score=95.0 - index,
            priority="high" if risk == "high" or index == 0 else "medium",
            category="data_quality",
            title=f"Resolve triage action {index + 1}",
            reason=f"Dataset triage risk is {risk}; readiness={triage_summary.get('readiness_score', '-')}/100.",
            action=str(action),
            source="dataset_triage",
            expected_signal="The triage readiness score should rise and blocker count should fall after cleanup.",
        )

    suggested_training = _training_suggestion(
        base_config,
        sample_count=sample_count,
        class_counts=class_counts,
        feature_separability_report=feature_separability_report or (dataset_triage_report or {}).get("feature_separability"),
        neighborhood_hardness_report=neighborhood_hardness_report or (dataset_triage_report or {}).get("neighborhood_hardness"),
    )
    if not metrics:
        add(
            score=90.0,
            priority="high",
            category="training",
            title="Train a baseline with the advisor-selected settings",
            reason="A dataset is loaded but no active validation metrics are available.",
            action="Run Train once or Run auto experiments with the suggested configuration.",
            source="model_state",
            suggested_config=suggested_training,
            expected_signal="A baseline run should produce validation F1, fixed-threshold metrics, Brier score, and ECE.",
        )
    else:
        _add_metric_recommendations(add, metrics, suggested_training)

    _add_report_recommendations(
        add,
        metrics=metrics,
        leakage_sentinel_report=leakage_sentinel_report or (dataset_triage_report or {}).get("leakage_sentinel"),
        threshold_report=threshold_report,
        calibration_repair_report=calibration_repair_report,
        calibration_slice_report=calibration_slice_report,
        external_holdout_report=external_holdout_report,
        decision_curve_report=decision_curve_report,
        selective_risk_report=selective_risk_report,
        stress_report=stress_report,
        permutation_null_report=permutation_null_report,
        population_drift_report=population_drift_report,
        adversarial_validation_report=adversarial_validation_report,
        chronological_holdout_report=chronological_holdout_report,
        learning_curve_report=learning_curve_report,
        data_acquisition_report=data_acquisition_report,
        data_value_report=data_value_report,
        prototype_audit_report=prototype_audit_report,
        ood_sentinel_report=ood_sentinel_report,
    )

    if len(trial_history) < 3 and metrics:
        add(
            score=48.0,
            priority="medium",
            category="search",
            title="Run a small auto-experiment sweep",
            reason=f"Only {len(trial_history)} trial record(s) are available, so model-choice evidence is thin.",
            action="Run auto experiments with 8-16 trials before comparing backends or saving a final model.",
            source="trial_history",
            suggested_config={**suggested_training, "trials": max(8, int(suggested_training.get("trials", 8)))},
            expected_signal="The trial history should show whether a feature map or backend is consistently better.",
        )

    return _finalize(sample_count, input_dim, class_counts, metrics, trial_history, recommendations, max_recommendations)


def format_experiment_advisor_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top = report.get("recommendations", [{}])[0] if report.get("recommendations") else {}
    return (
        "Experiment advisor: "
        f"recommendations={int(summary.get('recommendation_count', 0))}, "
        f"top={top.get('priority', '-')}/{top.get('category', '-')}, "
        f"next={summary.get('recommended_next_step') or top.get('title', 'none')}"
    )


def _training_suggestion(
    base_config: dict[str, Any],
    *,
    sample_count: int,
    class_counts: dict[str, int] | None,
    feature_separability_report: dict[str, Any] | None,
    neighborhood_hardness_report: dict[str, Any] | None,
) -> dict[str, Any]:
    suggestion = dict(base_config)
    separability = (feature_separability_report or {}).get("summary", {})
    hardness = (neighborhood_hardness_report or {}).get("summary", {})
    input_dim = int((feature_separability_report or {}).get("input_dim", 0))
    weak_features = int(separability.get("weak_feature_count", 0))
    strong_features = int(separability.get("strong_feature_count", 0))
    near_perfect = int(separability.get("near_perfect_feature_count", 0))
    loo_accuracy = float(hardness.get("loo_accuracy", 1.0))

    if near_perfect:
        suggestion["feature_map"] = "linear"
        suggestion["l1_penalty"] = max(float(suggestion.get("l1_penalty") or 0.0), 0.001)
    elif input_dim and weak_features == input_dim:
        suggestion["feature_map"] = "rff"
        suggestion["rff_components"] = max(int(suggestion.get("rff_components") or 64), 96)
    elif strong_features == 0 and loo_accuracy < 0.80:
        suggestion["feature_map"] = "quadratic"

    if sample_count < 80:
        suggestion["use_cv"] = True
        suggestion["kfold_splits"] = 5 if sample_count >= 20 else 3
    if class_counts:
        minority = min(class_counts.values())
        majority = max(class_counts.values())
        if majority / max(minority, 1) >= 3.0:
            suggestion["use_smote"] = True
            suggestion["smote_k"] = min(3, max(1, minority - 1))
    suggestion["trials"] = max(8, int(suggestion.get("trials", 8) or 8))
    suggestion["max_epochs"] = max(50, int(suggestion.get("max_epochs", 50) or 50))
    return suggestion


def _add_metric_recommendations(
    add: Any,
    metrics: dict[str, float | int],
    suggested_training: dict[str, Any],
) -> None:
    f1 = _metric(metrics, "f1")
    fixed_f1 = _metric(metrics, "fixed_threshold_f1")
    threshold_gain = _metric(metrics, "threshold_gain_f1", f1 - fixed_f1)
    ece = _metric(metrics, "ece")
    brier = _metric(metrics, "brier_score")
    recall = _metric(metrics, "recall")
    precision = _metric(metrics, "precision")

    if f1 < 0.60:
        add(
            score=86.0,
            priority="high",
            category="model_selection",
            title="Escalate model search before trusting this boundary",
            reason=f"Validation F1 is {f1:.3f}, below the practical 0.60 checkpoint.",
            action="Run auto experiments with the suggested feature map, then compare NumPy/MPS/Keras backends.",
            source="metrics",
            suggested_config={**suggested_training, "trials": max(12, int(suggested_training.get("trials", 8)))},
            expected_signal="The best trial should improve validation F1 without worsening Brier/ECE sharply.",
        )
    if threshold_gain >= 0.08:
        add(
            score=76.0,
            priority="high",
            category="thresholding",
            title="Promote threshold tuning to a first-class experiment",
            reason=f"Tuned-threshold F1 beats fixed 0.5 F1 by {threshold_gain:.3f}.",
            action="Run Threshold tradeoff and Decision curve, then choose an operating point from costs.",
            source="metrics",
            expected_signal="The selected threshold should match the target precision/recall or utility tradeoff.",
        )
    if ece >= 0.08 or brier >= 0.22:
        add(
            score=70.0,
            priority="medium",
            category="calibration",
            title="Check probability calibration before using probabilities operationally",
            reason=f"ECE={ece:.3f}, Brier={brier:.3f}; probabilities may be poorly calibrated.",
            action="Run Calibration repair and Post-hoc conformal diagnostics on held-out or reviewed rows.",
            source="metrics",
            expected_signal="A repair method should reduce Brier/ECE on evaluation rows, not just calibration rows.",
        )
    if recall < 0.55 and precision > recall + 0.15:
        add(
            score=66.0,
            priority="medium",
            category="thresholding",
            title="Explore a higher-recall operating point",
            reason=f"Recall={recall:.3f} is much lower than precision={precision:.3f}.",
            action="Run Threshold tradeoff with a recall target and compare the false-positive cost.",
            source="metrics",
            expected_signal="A lower threshold should raise recall while keeping precision acceptable for the task.",
        )


def _add_report_recommendations(
    add: Any,
    *,
    metrics: dict[str, float | int],
    leakage_sentinel_report: dict[str, Any] | None,
    threshold_report: dict[str, Any] | None,
    calibration_repair_report: dict[str, Any] | None,
    calibration_slice_report: dict[str, Any] | None,
    external_holdout_report: dict[str, Any] | None,
    decision_curve_report: dict[str, Any] | None,
    selective_risk_report: dict[str, Any] | None,
    stress_report: dict[str, Any] | None,
    permutation_null_report: dict[str, Any] | None,
    population_drift_report: dict[str, Any] | None,
    adversarial_validation_report: dict[str, Any] | None,
    chronological_holdout_report: dict[str, Any] | None,
    learning_curve_report: dict[str, Any] | None,
    data_acquisition_report: dict[str, Any] | None,
    data_value_report: dict[str, Any] | None,
    prototype_audit_report: dict[str, Any] | None,
    ood_sentinel_report: dict[str, Any] | None,
) -> None:
    if leakage_sentinel_report:
        summary = leakage_sentinel_report.get("summary", {})
        risk = str(summary.get("risk_level", "low"))
        top = summary.get("top_feature")
        top_text = "-" if top is None else f"x{int(top) + 1}"
        score = float(summary.get("max_risk_score", 0.0) or 0.0)
        if risk == "high":
            add(
                score=97.0,
                priority="high",
                category="leakage",
                title="Quarantine possible target leakage before more training",
                reason=f"Leakage sentinel flags {top_text} with risk score {score:.3f}.",
                action=str(
                    summary.get("recommendation")
                    or "Remove or quarantine the flagged feature, then rerun triage and validation."
                ),
                source="leakage_sentinel",
                expected_signal="After removal, validation F1 should remain plausible and external/holdout evidence should not collapse.",
            )
        elif risk == "medium":
            add(
                score=69.0,
                priority="medium",
                category="leakage",
                title="Review possible proxy shortcut",
                reason=f"Leakage sentinel marks {top_text} for proxy review.",
                action=str(summary.get("recommendation") or "Confirm the feature is available at prediction time."),
                source="leakage_sentinel",
            )
    if metrics and not external_holdout_report:
        add(
            score=54.0,
            priority="medium",
            category="external_validation",
            title="Run an external holdout check before final save",
            reason="The active model has validation metrics, but no separate labeled CSV has been scored as external evidence.",
            action="Use Evaluate holdout with a labeled CSV that was not used for training, tuning, or threshold selection.",
            source="missing_external_holdout",
            expected_signal="External holdout F1, calibration, and shift checks should agree with the internal validation story.",
            )
    if external_holdout_report:
        summary = external_holdout_report.get("summary", {})
        verdict = str(summary.get("verdict", "holdout_performance_review"))
        f1 = float(summary.get("f1", 0.0) or 0.0)
        if verdict == "holdout_failure":
            add(
                score=98.0,
                priority="high",
                category="external_validation",
                title="Stop and investigate external holdout failure",
                reason=f"The external holdout verdict is failure with F1={f1:.3f}.",
                action=str(summary.get("recommendation") or "Inspect holdout errors before saving or promoting this model."),
                source="external_holdout",
                expected_signal="After remediation, holdout F1 and balanced accuracy should recover without retuning on the holdout.",
            )
        elif verdict != "holdout_pass":
            add(
                score=72.0,
                priority="medium",
                category="external_validation",
                title="Review external holdout evidence",
                reason=f"The external holdout verdict is {verdict}.",
                action=str(summary.get("recommendation") or "Compare holdout errors, calibration, and shift against internal validation."),
                source="external_holdout",
                expected_signal="Holdout evidence should either confirm deployment readiness or identify the dataset shift to resolve.",
            )
    if data_acquisition_report:
        summary = data_acquisition_report.get("summary", {})
        priority = str(summary.get("priority", "low"))
        budget = int(summary.get("recommended_label_budget", 0) or 0)
        next_step = str(summary.get("recommended_next_step") or "Follow the data acquisition plan.")
        if priority == "high":
            add(
                score=89.0,
                priority="high",
                category="data_collection",
                title="Follow the high-priority data acquisition plan",
                reason=f"The planner recommends {budget} label(s) before the next serious model-selection pass.",
                action=next_step,
                source="data_acquisition_plan",
                expected_signal="After collecting labels, planner priority should fall and validation evidence should become less fragile.",
            )
        elif priority == "medium":
            add(
                score=63.0,
                priority="medium",
                category="data_collection",
                title="Target the planner's boundary or coverage gaps",
                reason=f"The planner found targeted acquisition gaps with a suggested budget of {budget}.",
                action=next_step,
                source="data_acquisition_plan",
                expected_signal="Boundary/tail candidate counts should fall or become documented after review.",
            )
    if data_value_report:
        summary = data_value_report.get("summary", {})
        priority = str(summary.get("priority", "low"))
        review_count = int(summary.get("review_row_count", 0) or 0)
        redundant_count = int(summary.get("redundant_row_count", 0) or 0)
        next_step = str(summary.get("recommended_next_step") or "Review the data value scout rows.")
        if priority == "high" or review_count >= 3:
            add(
                score=88.0,
                priority="high",
                category="data_curation",
                title="Curate high-impact rows before more model search",
                reason=f"Data value scout found {review_count} review row(s) that may harm neighborhood evidence.",
                action=next_step,
                source="data_value_scout",
                expected_signal="After curation, review-row count should fall and validation evidence should become less brittle.",
            )
        elif priority == "medium" or redundant_count:
            add(
                score=61.0,
                priority="medium",
                category="data_curation",
                title="Review row-value and redundancy findings",
                reason=f"Data value scout found {review_count} review row(s) and {redundant_count} redundant anchor row(s).",
                action=next_step,
                source="data_value_scout",
                expected_signal="Dataset curation should reduce redundant weight or document rare rows as canaries.",
            )
    if learning_curve_report:
        summary = learning_curve_report.get("summary", {})
        verdict = str(summary.get("verdict", "stable_enough"))
        final_f1 = float(summary.get("final_f1", summary.get("best_f1", 0.0)) or 0.0)
        gain = float(summary.get("f1_gain", 0.0) or 0.0)
        action = str(summary.get("recommended_next_step") or "Review learning-curve diagnostics before the next run.")
        if verdict == "underfit_or_noisy":
            add(
                score=84.0,
                priority="high",
                category="data_quality",
                title="Resolve weak learning-curve signal",
                reason=f"Fixed-holdout learning curve ends at F1={final_f1:.3f}, below the practical checkpoint.",
                action=action,
                source="learning_curve",
                expected_signal="Later runs should show a higher final fixed-holdout F1 before promotion.",
            )
        elif verdict == "more_data_helpful":
            add(
                score=68.0,
                priority="medium",
                category="data_volume",
                title="Collect labels where the learning curve is still rising",
                reason=f"Fixed-holdout F1 improved by {gain:.3f} as training size increased.",
                action=action,
                source="learning_curve",
                expected_signal="After adding reviewed labels, the final point should rise or the curve should flatten.",
            )
        elif verdict == "capacity_or_regularization_review":
            add(
                score=60.0,
                priority="medium",
                category="model_selection",
                title="Review capacity or regularization against the learning curve",
                reason="The best learning-curve point did not come from the largest training fraction.",
                action=action,
                source="learning_curve",
                expected_signal="Reruns should reduce the gap between the best point and the full-training point.",
            )
    if calibration_slice_report:
        summary = calibration_slice_report.get("summary", {})
        risk = str(summary.get("risk_level", "low"))
        worst = str(summary.get("worst_slice", "none"))
        gap = float(summary.get("max_absolute_confidence_gap", 0.0) or 0.0)
        if risk == "high":
            add(
                score=78.0,
                priority="high",
                category="calibration",
                title="Repair or explain localized calibration risk",
                reason=f"Calibration slices flag {worst} with confidence gap {gap:.3f}.",
                action=str(summary.get("recommendation") or "Run calibration repair and inspect the worst local slice."),
                source="calibration_slices",
                expected_signal="The worst local confidence gap should shrink or be documented as a known limitation.",
            )
        elif risk == "medium":
            add(
                score=59.0,
                priority="medium",
                category="calibration",
                title="Inspect local probability trust before deployment",
                reason=f"Calibration slices flag {worst} for review.",
                action=str(summary.get("recommendation") or "Inspect the worst local calibration slice before using probabilities."),
                source="calibration_slices",
            )
    if calibration_repair_report:
        summary = calibration_repair_report.get("summary", {})
        if float(summary.get("best_brier_improvement", 0.0)) >= 0.03 or float(summary.get("best_ece_improvement", 0.0)) >= 0.03:
            add(
                score=58.0,
                priority="medium",
                category="calibration",
                title="Compare the recommended calibration repair",
                reason=f"Calibration repair recommends {summary.get('recommended_method', 'a method')} with measurable improvement.",
                action="Export the report and rerun predictions with calibration repair evidence in the model card.",
                source="calibration_repair",
                expected_signal="Brier/ECE should improve on held-out rows without hiding poor threshold metrics.",
            )
    if stress_report and float(stress_report.get("summary", {}).get("stress_f1_ratio", 1.0)) < 0.80:
        add(
            score=62.0,
            priority="medium",
            category="robustness",
            title="Investigate robustness before saving this model",
            reason="The robustness stress lab shows a large F1 drop under perturbation.",
            action="Inspect the worst perturbation, add representative rows, or reduce reliance on brittle features.",
            source="stress_lab",
        )
    drift_sources = [
        (population_drift_report, "population_drift", "Population drift suggests current rows differ from reference rows."),
        (adversarial_validation_report, "adversarial_validation", "Adversarial validation says row groups are distinguishable."),
        (chronological_holdout_report, "chronological_holdout", "Chronological holdout suggests the later rule may have degraded."),
    ]
    for report, source, reason in drift_sources:
        verdict = str((report or {}).get("summary", {}).get("verdict", "")).lower()
        if any(token in verdict for token in ("strong", "severe", "degradation")):
            add(
                score=64.0,
                priority="medium",
                category="validation",
                title=f"Follow up {source.replace('_', ' ')} evidence",
                reason=reason,
                action="Use a held-out, reviewed, or later-slice validation set before calling the model stable.",
                source=source,
            )
    permutation_verdict = str((permutation_null_report or {}).get("summary", {}).get("verdict", "")).lower()
    if any(token in permutation_verdict for token in ("weak", "noise", "null")):
        add(
            score=64.0,
            priority="medium",
            category="validation",
            title="Follow up weak permutation-null evidence",
            reason="Permutation-null evidence is weak for the current validation score.",
            action="Use a held-out, reviewed, or later-slice validation set before calling the model stable.",
            source="permutation_null",
        )
    if selective_risk_report and float(selective_risk_report.get("summary", {}).get("min_selective_risk", 1.0)) < 0.05:
        add(
            score=52.0,
            priority="low",
            category="abstention",
            title="Consider an abstention policy",
            reason="Selective-risk diagnostics found a low-risk subset after abstaining on uncertain rows.",
            action="Choose a confidence cutoff only after checking coverage and subgroup/slice effects.",
            source="selective_risk",
        )
    if threshold_report is None and decision_curve_report is None:
        add(
            score=46.0,
            priority="low",
            category="thresholding",
            title="Run operating-point diagnostics",
            reason="No threshold or decision-curve report is available yet.",
            action="Run Threshold tradeoff and Decision curve after the next trained model.",
            source="missing_diagnostics",
        )
    if prototype_audit_report and int(prototype_audit_report.get("summary", {}).get("label_contradiction_count", 0)) > 0:
        add(
            score=57.0,
            priority="medium",
            category="data_quality",
            title="Review local label contradictions before another tuning sweep",
            reason="Prototype audit found rows with opposite-label neighborhoods.",
            action="Inspect the ranked contradiction rows and fix labels or add disambiguating features.",
            source="prototype_audit",
        )
    if ood_sentinel_report and int(ood_sentinel_report.get("summary", {}).get("flagged_count", 0)) > 0:
        add(
            score=50.0,
            priority="low",
            category="data_quality",
            title="Inspect OOD sentinel rows",
            reason="The OOD sentinel found rows above its review threshold.",
            action="Check whether flagged rows are valid edge cases, artifacts, or import mistakes.",
            source="ood_sentinel",
        )


def _finalize(
    sample_count: int,
    input_dim: int | None,
    class_counts: dict[str, int] | None,
    metrics: dict[str, float | int],
    trial_history: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    max_recommendations: int,
) -> dict[str, Any]:
    recommendations.sort(key=lambda item: (-float(item["priority_score"]), item["category"], item["title"]))
    limited = []
    seen: set[tuple[str, str]] = set()
    for item in recommendations:
        key = (str(item["category"]), str(item["title"]))
        if key in seen:
            continue
        seen.add(key)
        row = dict(item)
        row["rank"] = len(limited) + 1
        limited.append(row)
        if len(limited) >= max_recommendations:
            break
    top = limited[0] if limited else {}
    return {
        "sample_count": int(sample_count),
        "input_dim": input_dim,
        "class_counts": class_counts,
        "metric_snapshot": dict(metrics),
        "trial_count": len(trial_history),
        "summary": {
            "recommendation_count": len(limited),
            "top_priority": top.get("priority"),
            "top_category": top.get("category"),
            "recommended_next_step": top.get("title"),
            "needs_training": not bool(metrics),
            "model_f1": metrics.get("f1"),
        },
        "recommendations": limited,
    }


def _labels_to_array(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        return np.asarray(labels, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.int32)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))}


def _metric(metrics: dict[str, float | int], key: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value
