from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .audit import audit_dataset
from .data import validate_dataset
from .feature_separability import run_feature_separability_diagnostics
from .neighborhood_hardness import run_neighborhood_hardness_diagnostics
from .ood_sentinel import run_ood_sentinel
from .prototype_audit import run_prototype_audit


def run_dataset_triage(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Run model-free dataset diagnostics and summarize readiness for training."""
    dataset = validate_dataset(
        _sequence_to_lists(features),
        _sequence_to_lists(labels),
        min_samples=6,
        require_two_classes=True,
    )
    class_counts = {
        "0": int(np.sum(dataset.labels == 0)),
        "1": int(np.sum(dataset.labels == 1)),
    }
    if min(class_counts.values()) < 2:
        raise ValueError("Dataset triage needs at least two rows per class.")

    dataset_audit = audit_dataset(dataset.features, dataset.labels)
    feature_separability = run_feature_separability_diagnostics(dataset.features, dataset.labels)
    prototype_audit = run_prototype_audit(dataset.features, dataset.labels)
    neighborhood_hardness = run_neighborhood_hardness_diagnostics(dataset.features, dataset.labels)
    ood_sentinel = run_ood_sentinel(None, dataset.features, dataset.labels)
    summary = _triage_summary(
        dataset_audit=dataset_audit,
        feature_separability=feature_separability,
        prototype_audit=prototype_audit,
        neighborhood_hardness=neighborhood_hardness,
        ood_sentinel=ood_sentinel,
    )
    return {
        "sample_count": dataset.sample_count,
        "input_dim": dataset.input_dim,
        "class_counts": class_counts,
        "summary": summary,
        "dataset_audit": dataset_audit,
        "feature_separability": feature_separability,
        "prototype_audit": prototype_audit,
        "neighborhood_hardness": neighborhood_hardness,
        "ood_sentinel": ood_sentinel,
    }


def _sequence_to_lists(value: Sequence[Any] | np.ndarray) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def format_dataset_triage_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    actions = summary.get("top_actions") or []
    action_text = "; ".join(str(action) for action in actions[:3]) if actions else "none"
    return (
        "Dataset triage: "
        f"readiness={float(summary.get('readiness_score', 0.0)):.1f}/100, "
        f"risk={summary.get('risk_level', '-')}, "
        f"blockers={int(summary.get('blocking_issue_count', 0))}, "
        f"actions={action_text}"
    )


def _triage_summary(
    *,
    dataset_audit: dict[str, Any],
    feature_separability: dict[str, Any],
    prototype_audit: dict[str, Any],
    neighborhood_hardness: dict[str, Any],
    ood_sentinel: dict[str, Any],
) -> dict[str, Any]:
    penalties: list[tuple[float, str, bool]] = []

    def add(points: float, action: str, *, blocking: bool = False) -> None:
        if points > 0.0:
            penalties.append((float(points), action, blocking))

    audit_warnings = set(str(item) for item in dataset_audit.get("warnings", []))
    sample_count = int(dataset_audit.get("sample_count", 0))
    class_balance = dataset_audit.get("class_balance", {})
    minority_count = int(class_balance.get("minority_count", 0))
    imbalance_ratio = float(dataset_audit.get("imbalance_ratio", 1.0))
    duplicate_count = int(dataset_audit.get("duplicate_row_count", 0))
    conflict_count = int(dataset_audit.get("label_conflict_count", 0))
    conflicting_rows = int(dataset_audit.get("conflicting_row_count", 0))
    constant_count = len(dataset_audit.get("constant_features", []))
    high_correlation_count = len(dataset_audit.get("high_correlations", []))

    if sample_count < 20 or "very small dataset" in audit_warnings:
        add(10.0, "Collect more rows before trusting validation metrics.")
    if minority_count < 6:
        add(8.0, "Add more minority-class examples or keep class-weighted training enabled.")
    if imbalance_ratio >= 4.0:
        add(12.0, "Balance the dataset or use imbalance-aware metrics and thresholds.")
    if duplicate_count:
        add(min(12.0, 2.0 * duplicate_count), "Deduplicate repeated feature rows before tuning.")
    if conflict_count or conflicting_rows:
        add(
            min(28.0, 18.0 + 2.0 * max(conflict_count, conflicting_rows)),
            "Review same-feature rows with conflicting labels.",
            blocking=True,
        )
    if constant_count:
        add(min(10.0, 3.0 * constant_count), "Remove or explain constant feature columns.")
    if high_correlation_count:
        add(min(8.0, 2.0 * high_correlation_count), "Inspect highly correlated features for redundancy.")

    separability_summary = feature_separability.get("summary", {})
    near_perfect = int(separability_summary.get("near_perfect_feature_count", 0))
    weak_features = int(separability_summary.get("weak_feature_count", 0))
    input_dim = int(feature_separability.get("input_dim", 0))
    strong_features = int(separability_summary.get("strong_feature_count", 0))
    redundant_pairs = int(separability_summary.get("redundant_pair_count", 0))
    if near_perfect:
        add(12.0, "Inspect near-perfect single features for leakage or shortcut coding.", blocking=near_perfect >= 2)
    if input_dim and weak_features == input_dim:
        add(14.0, "Add better features or try nonlinear feature maps; all single-feature scans look weak.")
    elif strong_features == 0:
        add(6.0, "Expect multivariate or nonlinear modeling; no strong one-feature separator was found.")
    if redundant_pairs:
        add(min(8.0, 2.0 * redundant_pairs), "Simplify redundant features or keep regularization active.")

    prototype_summary = prototype_audit.get("summary", {})
    contradiction_count = int(prototype_summary.get("label_contradiction_count", 0))
    isolated_count = int(prototype_summary.get("isolated_row_count", 0))
    boundary_count = int(prototype_summary.get("boundary_row_count", 0))
    if contradiction_count:
        add(
            min(20.0, 5.0 * contradiction_count),
            "Review rows whose nearest neighbors mostly have the opposite label.",
            blocking=contradiction_count >= 2,
        )
    if isolated_count:
        add(min(10.0, 2.5 * isolated_count), "Collect coverage around sparse same-class islands.")
    if sample_count and boundary_count / sample_count >= 0.25:
        add(6.0, "Plan abstention or more boundary labels; many rows sit near class boundaries.")

    hardness_summary = neighborhood_hardness.get("summary", {})
    loo_accuracy = float(hardness_summary.get("loo_accuracy", 1.0))
    hard_count = int(hardness_summary.get("hard_row_count", 0))
    ambiguous_count = int(hardness_summary.get("ambiguous_row_count", 0))
    label_issue_count = int(hardness_summary.get("label_issue_candidate_count", 0))
    if loo_accuracy < 0.65:
        add(18.0, "Treat local labels as low-confidence; nearest-neighbor agreement is poor.", blocking=True)
    elif loo_accuracy < 0.80:
        add(8.0, "Inspect hard neighborhoods before trusting a high-capacity model.")
    if label_issue_count:
        add(
            min(22.0, 5.0 * label_issue_count),
            "Review label-issue candidates from local voting.",
            blocking=label_issue_count >= 2,
        )
    if sample_count and hard_count / sample_count >= 0.20:
        add(8.0, "Use sample review or active learning; many rows are locally hard.")
    if sample_count and ambiguous_count / sample_count >= 0.15:
        add(6.0, "Expect uncertain predictions around ambiguous neighborhoods.")

    ood_summary = ood_sentinel.get("summary", {})
    flagged_count = int(ood_summary.get("flagged_count", ood_summary.get("flagged_row_count", 0)))
    max_abs_z = float(ood_summary.get("max_abs_robust_z", 0.0))
    if flagged_count:
        add(min(12.0, 2.0 * flagged_count), "Inspect OOD-sentinel rows for artifacts, leverage, or data-entry issues.")
    if max_abs_z >= 6.0:
        add(6.0, "Check extreme feature tails before using them for model selection.")

    score = max(0.0, 100.0 - sum(points for points, _, _ in penalties))
    blocking_count = int(sum(1 for _, _, blocking in penalties if blocking))
    if score < 55.0 or blocking_count >= 2:
        risk_level = "high"
    elif score < 78.0 or blocking_count:
        risk_level = "medium"
    else:
        risk_level = "low"

    top_actions = _dedupe_actions(penalties)
    warning = "; ".join(action for _, action, blocking in sorted(penalties, reverse=True) if blocking) or None
    return {
        "readiness_score": round(float(score), 1),
        "risk_level": risk_level,
        "blocking_issue_count": blocking_count,
        "top_actions": top_actions,
        "penalty_points": round(float(100.0 - score), 1),
        "warning": warning,
    }


def _dedupe_actions(penalties: list[tuple[float, str, bool]], limit: int = 5) -> list[str]:
    best_by_action: dict[str, tuple[float, bool]] = {}
    for points, action, blocking in penalties:
        previous = best_by_action.get(action)
        if previous is None or (blocking, points) > (previous[1], previous[0]):
            best_by_action[action] = (points, blocking)
    ranked = sorted(
        ((points, blocking, action) for action, (points, blocking) in best_by_action.items()),
        key=lambda item: (not item[1], -item[0], item[2]),
    )
    return [action for _, _, action in ranked[:limit]]
