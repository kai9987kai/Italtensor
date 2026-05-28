from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import ModelConfig


def build_promotion_gate(
    *,
    sample_count: int,
    input_dim: int | None,
    labels: Sequence[int] | np.ndarray,
    config: ModelConfig | None = None,
    metrics: dict[str, float | int] | None = None,
    trial_history: list[dict[str, Any]] | None = None,
    dataset_triage_report: dict[str, Any] | None = None,
    experiment_advisor_report: dict[str, Any] | None = None,
    trial_inspector_report: dict[str, Any] | None = None,
    threshold_report: dict[str, Any] | None = None,
    calibration_repair_report: dict[str, Any] | None = None,
    stress_report: dict[str, Any] | None = None,
    permutation_null_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    adversarial_validation_report: dict[str, Any] | None = None,
    chronological_holdout_report: dict[str, Any] | None = None,
    selective_risk_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transparent model-promotion checklist from local experiment evidence."""
    metrics = metrics or {}
    trial_history = trial_history or []
    labels_array = _labels_to_array(labels)
    class_counts = _class_counts(labels_array) if labels_array.size else None
    checks: list[dict[str, Any]] = []

    def add(
        *,
        severity: str,
        category: str,
        title: str,
        status: str,
        evidence: str,
        action: str,
        penalty: float,
    ) -> None:
        checks.append(
            {
                "severity": severity,
                "category": category,
                "title": title,
                "status": status,
                "evidence": evidence,
                "action": action,
                "penalty": float(max(0.0, penalty)),
            }
        )

    if sample_count <= 0 or labels_array.size == 0:
        add(
            severity="blocker",
            category="data",
            title="Dataset evidence is missing",
            status="fail",
            evidence="No loaded labels are available for a promotion report.",
            action="Load the dataset used for validation before saving a final model.",
            penalty=35.0,
        )
    else:
        if sample_count < 12:
            add(
                severity="blocker",
                category="data",
                title="Dataset is too small for promotion",
                status="fail",
                evidence=f"Only {sample_count} labeled row(s) are loaded.",
                action="Collect more labeled rows or treat the model as a prototype only.",
                penalty=24.0,
            )
        elif sample_count < 40:
            add(
                severity="caution",
                category="data",
                title="Validation evidence is thin",
                status="review",
                evidence=f"{sample_count} labeled row(s) are loaded.",
                action="Prefer cross-validation or a fresh reviewed holdout before promotion.",
                penalty=8.0,
            )
        if class_counts:
            minority = min(class_counts.values())
            majority = max(class_counts.values())
            if minority < 3:
                add(
                    severity="blocker",
                    category="data",
                    title="Minority class evidence is insufficient",
                    status="fail",
                    evidence=f"Class counts are {class_counts}.",
                    action="Add more minority-class examples before trusting validation metrics.",
                    penalty=24.0,
                )
            elif majority / max(minority, 1) >= 4.0:
                add(
                    severity="caution",
                    category="data",
                    title="Class imbalance needs an operating-point note",
                    status="review",
                    evidence=f"Class counts are {class_counts}.",
                    action="Use balanced metrics, threshold tradeoffs, and the selected threshold in the release note.",
                    penalty=7.0,
                )

    f1 = _metric(metrics, "f1")
    accuracy = _metric(metrics, "accuracy")
    balanced_accuracy = _metric(metrics, "balanced_accuracy", fallback=accuracy)
    brier = _metric(metrics, "brier_score")
    ece = _metric(metrics, "ece")
    threshold_gain = _metric(metrics, "threshold_gain_f1", fallback=0.0)
    fixed_f1 = _metric(metrics, "fixed_threshold_f1")
    if not metrics:
        add(
            severity="blocker",
            category="model",
            title="Validation metrics are missing",
            status="fail",
            evidence="No active model validation metrics are available.",
            action="Train or load a model with validation metrics before promotion.",
            penalty=35.0,
        )
    else:
        if f1 < 0.60 or balanced_accuracy < 0.60:
            add(
                severity="blocker",
                category="model",
                title="Primary validation score is below promotion floor",
                status="fail",
                evidence=f"F1={f1:.3f}, balanced_accuracy={balanced_accuracy:.3f}.",
                action="Improve data quality or model selection before saving this as final.",
                penalty=28.0,
            )
        elif f1 < 0.75:
            add(
                severity="caution",
                category="model",
                title="Primary score needs review",
                status="review",
                evidence=f"F1={f1:.3f}.",
                action="Promote only with a documented tolerance for this error profile.",
                penalty=10.0,
            )
        if accuracy - f1 >= 0.18:
            add(
                severity="caution",
                category="model",
                title="Accuracy may be hiding class errors",
                status="review",
                evidence=f"Accuracy={accuracy:.3f}, F1={f1:.3f}.",
                action="Use precision, recall, balanced accuracy, and threshold tradeoff evidence.",
                penalty=8.0,
            )
        if ece >= 0.12 or brier >= 0.28:
            add(
                severity="blocker",
                category="calibration",
                title="Probability calibration is poor",
                status="fail",
                evidence=f"ECE={ece:.3f}, Brier={brier:.3f}.",
                action="Run calibration repair or avoid operational probability decisions.",
                penalty=18.0,
            )
        elif ece >= 0.08 or brier >= 0.22:
            add(
                severity="caution",
                category="calibration",
                title="Probability calibration needs review",
                status="review",
                evidence=f"ECE={ece:.3f}, Brier={brier:.3f}.",
                action="Run calibration repair and include raw vs repaired Brier/ECE in the report.",
                penalty=8.0,
            )
        if threshold_gain >= 0.08 and not threshold_report:
            add(
                severity="caution",
                category="threshold",
                title="Chosen threshold is material to performance",
                status="review",
                evidence=f"Tuned-threshold F1 gain is {threshold_gain:.3f}; fixed-threshold F1={fixed_f1:.3f}.",
                action="Run Threshold tradeoff and document the selected operating point.",
                penalty=7.0,
            )

    _add_triage_checks(add, dataset_triage_report)
    _add_trial_checks(add, trial_history, trial_inspector_report)
    _add_experiment_advisor_checks(add, experiment_advisor_report)
    _add_repair_and_robustness_checks(
        add,
        calibration_repair_report=calibration_repair_report,
        stress_report=stress_report,
        permutation_null_report=permutation_null_report,
        population_drift_report=population_drift_report,
        adversarial_validation_report=adversarial_validation_report,
        chronological_holdout_report=chronological_holdout_report,
        selective_risk_report=selective_risk_report,
    )

    checks = _rank_checks(checks)
    score = max(0.0, 100.0 - sum(float(item["penalty"]) for item in checks))
    blocker_count = sum(1 for item in checks if item["severity"] == "blocker")
    caution_count = sum(1 for item in checks if item["severity"] == "caution")
    if blocker_count or score < 60.0:
        verdict = "blocked"
    elif caution_count or score < 82.0:
        verdict = "needs_review"
    else:
        verdict = "promotable"
    top_action = checks[0]["action"] if checks else "Save the model with this promotion report and trial evidence."
    return {
        "sample_count": int(sample_count),
        "input_dim": input_dim,
        "class_counts": class_counts,
        "model_config": config.to_dict() if config is not None else None,
        "metrics": metrics,
        "summary": {
            "promotion_score": round(float(score), 1),
            "verdict": verdict,
            "blocker_count": int(blocker_count),
            "caution_count": int(caution_count),
            "best_f1": _round_or_none(f1 if metrics else None),
            "accuracy": _round_or_none(accuracy if metrics else None),
            "balanced_accuracy": _round_or_none(balanced_accuracy if metrics else None),
            "brier_score": _round_or_none(brier if metrics else None),
            "ece": _round_or_none(ece if metrics else None),
            "required_next_step": top_action,
            "warning": _warning(verdict, blocker_count, caution_count),
        },
        "checks": checks,
        "release_note": {
            "recommended_use": "prototype" if verdict == "blocked" else ("guarded_local_use" if verdict == "needs_review" else "local_candidate"),
            "must_include": _must_include(checks, metrics, trial_history),
        },
    }


def format_promotion_gate_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Promotion gate: "
        f"verdict={summary.get('verdict', '-')}, "
        f"score={float(summary.get('promotion_score', 0.0)):.1f}/100, "
        f"blockers={int(summary.get('blocker_count', 0))}, "
        f"cautions={int(summary.get('caution_count', 0))}, "
        f"next={summary.get('required_next_step') or 'none'}"
    )


def _add_triage_checks(add: Any, dataset_triage_report: dict[str, Any] | None) -> None:
    if not dataset_triage_report:
        add(
            severity="caution",
            category="data_quality",
            title="Dataset triage has not been run",
            status="missing",
            evidence="No dataset-triage report is attached.",
            action="Run Dataset triage before final promotion.",
            penalty=6.0,
        )
        return
    summary = dataset_triage_report.get("summary", {})
    risk = str(summary.get("risk_level", "unknown"))
    blockers = int(summary.get("blocking_issue_count", 0) or 0)
    readiness = float(summary.get("readiness_score", 100.0) or 0.0)
    if risk == "high" or blockers:
        add(
            severity="blocker",
            category="data_quality",
            title="Dataset triage reports blockers",
            status="fail",
            evidence=f"risk={risk}, readiness={readiness:.1f}, blockers={blockers}.",
            action=(summary.get("top_actions") or ["Resolve dataset triage blockers before promotion."])[0],
            penalty=26.0,
        )
    elif risk == "medium" or readiness < 80.0:
        add(
            severity="caution",
            category="data_quality",
            title="Dataset triage needs review",
            status="review",
            evidence=f"risk={risk}, readiness={readiness:.1f}.",
            action=(summary.get("top_actions") or ["Review dataset triage findings before promotion."])[0],
            penalty=8.0,
        )


def _add_trial_checks(add: Any, trial_history: list[dict[str, Any]], trial_inspector_report: dict[str, Any] | None) -> None:
    valid_count = int((trial_inspector_report or {}).get("valid_trial_count", len(trial_history)) or 0)
    if valid_count < 3:
        add(
            severity="caution",
            category="model_selection",
            title="Model-selection evidence is thin",
            status="review",
            evidence=f"{valid_count} comparable trial(s) are available.",
            action="Run at least a small auto-experiment sweep before final promotion.",
            penalty=8.0,
        )
    if not trial_inspector_report:
        add(
            severity="caution",
            category="model_selection",
            title="Trial inspector has not been run",
            status="missing",
            evidence="No trial-inspector report is attached.",
            action="Run Trial inspector after training to document the leaderboard margin.",
            penalty=5.0,
        )
        return
    summary = trial_inspector_report.get("summary", {})
    margin = _optional_float(summary.get("leader_margin_f1"))
    if margin is not None and margin <= 0.02 and valid_count >= 2:
        add(
            severity="caution",
            category="model_selection",
            title="Leaderboard winner is unstable",
            status="review",
            evidence=f"Best F1 margin is {margin:.3f}.",
            action="Rerun the top settings with a new seed or cross-validation before promotion.",
            penalty=9.0,
        )


def _add_experiment_advisor_checks(add: Any, experiment_advisor_report: dict[str, Any] | None) -> None:
    if not experiment_advisor_report:
        return
    recs = experiment_advisor_report.get("recommendations") or []
    high = [item for item in recs if item.get("priority") == "high"]
    if high:
        top = high[0]
        add(
            severity="caution",
            category="open_recommendations",
            title="High-priority advisor item remains open",
            status="review",
            evidence=f"{top.get('category', '-')}: {top.get('title', '-')}.",
            action=str(top.get("action") or "Review high-priority advisor recommendations before promotion."),
            penalty=6.0,
        )


def _add_repair_and_robustness_checks(
    add: Any,
    *,
    calibration_repair_report: dict[str, Any] | None,
    stress_report: dict[str, Any] | None,
    permutation_null_report: dict[str, Any] | None,
    population_drift_report: dict[str, Any] | None,
    adversarial_validation_report: dict[str, Any] | None,
    chronological_holdout_report: dict[str, Any] | None,
    selective_risk_report: dict[str, Any] | None,
) -> None:
    if calibration_repair_report:
        summary = calibration_repair_report.get("summary", {})
        improvement = float(summary.get("best_brier_improvement", summary.get("brier_improvement", 0.0)) or 0.0)
        if improvement >= 0.03:
            add(
                severity="caution",
                category="calibration",
                title="Calibration repair materially improves probabilities",
                status="review",
                evidence=f"Best Brier improvement is {improvement:.3f}.",
                action="Save the raw model only with a note that repaired probabilities performed better.",
                penalty=5.0,
            )
    if stress_report:
        summary = stress_report.get("summary", {})
        ratio = float(summary.get("stress_f1_ratio", 1.0) or 1.0)
        worst_f1 = float(summary.get("worst_f1", 1.0) or 1.0)
        if ratio < 0.50 or worst_f1 < 0.45:
            add(
                severity="blocker",
                category="robustness",
                title="Stress lab shows severe fragility",
                status="fail",
                evidence=f"worst_f1={worst_f1:.3f}, stress_f1_ratio={ratio:.3f}.",
                action="Do not promote until the most damaging stress case is understood.",
                penalty=18.0,
            )
        elif ratio < 0.75:
            add(
                severity="caution",
                category="robustness",
                title="Stress lab shows material fragility",
                status="review",
                evidence=f"stress_f1_ratio={ratio:.3f}.",
                action="Document the worst perturbation and add monitoring or feature validation.",
                penalty=8.0,
            )
    if permutation_null_report:
        verdict = str((permutation_null_report.get("summary") or {}).get("verdict", ""))
        if "no_signal" in verdict or "weak" in verdict:
            add(
                severity="blocker" if "no_signal" in verdict else "caution",
                category="significance",
                title="Permutation-null evidence is weak",
                status="fail" if "no_signal" in verdict else "review",
                evidence=f"Permutation-null verdict is {verdict}.",
                action="Collect more data or improve signal before promotion.",
                penalty=16.0 if "no_signal" in verdict else 6.0,
            )
    for report, category, title in (
        (population_drift_report, "drift", "Population drift is material"),
        (adversarial_validation_report, "drift", "Adversarial validation detects shift"),
        (chronological_holdout_report, "temporal", "Chronological holdout degrades"),
    ):
        verdict = str(((report or {}).get("summary") or {}).get("verdict", ""))
        if "strong" in verdict or "severe" in verdict:
            add(
                severity="caution",
                category=category,
                title=title,
                status="review",
                evidence=f"verdict={verdict}.",
                action="Treat this as a guarded model and validate on fresh current-distribution rows.",
                penalty=7.0,
            )
    if selective_risk_report:
        summary = selective_risk_report.get("summary", {})
        coverage = _optional_float(summary.get("recommended_coverage"))
        if coverage is not None and coverage < 0.65:
            add(
                severity="caution",
                category="abstention",
                title="Selective prediction needs low coverage",
                status="review",
                evidence=f"Recommended coverage is {coverage:.3f}.",
                action="Promote only if abstention/review capacity is operationally acceptable.",
                penalty=6.0,
            )


def _rank_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    severity_order = {"blocker": 0, "caution": 1, "info": 2}
    ranked = sorted(
        checks,
        key=lambda item: (
            severity_order.get(str(item.get("severity")), 3),
            -float(item.get("penalty") or 0.0),
            str(item.get("category") or ""),
            str(item.get("title") or ""),
        ),
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def _must_include(checks: list[dict[str, Any]], metrics: dict[str, float | int], trial_history: list[dict[str, Any]]) -> list[str]:
    items = ["validation metrics", "decision threshold", "dataset summary"]
    if trial_history:
        items.append("trial history")
    if any(check["category"] == "calibration" for check in checks) or "ece" in metrics or "brier_score" in metrics:
        items.append("calibration note")
    if any(check["category"] in {"drift", "temporal", "robustness"} for check in checks):
        items.append("deployment-risk note")
    if any(check["severity"] == "blocker" for check in checks):
        items.append("blocked-use warning")
    return items


def _warning(verdict: str, blocker_count: int, caution_count: int) -> str | None:
    if verdict == "blocked":
        return f"Resolve {blocker_count} blocker(s) before promotion."
    if verdict == "needs_review":
        return f"Review {caution_count} caution(s) before relying on this model."
    return None


def _labels_to_array(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        return np.asarray(labels, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.int32)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "0": int(np.sum(labels == 0)),
        "1": int(np.sum(labels == 1)),
    }


def _metric(metrics: dict[str, float | int], key: str, *, fallback: float = 0.0) -> float:
    return _optional_float(metrics.get(key), fallback=fallback)


def _optional_float(value: Any, *, fallback: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None if fallback is None else float(fallback)
    if not np.isfinite(parsed):
        return None if fallback is None else float(fallback)
    return parsed


def _round_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)
