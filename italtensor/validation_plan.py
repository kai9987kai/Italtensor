from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .data import validate_dataset


def run_validation_plan(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Build a model-free validation strategy recommendation for the loaded dataset."""
    dataset = validate_dataset(
        _sequence_to_lists(features),
        _sequence_to_lists(labels),
        min_samples=1,
        require_two_classes=False,
    )
    x = dataset.features.astype(np.float64)
    y = dataset.labels.astype(np.int32)
    class_counts = {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}
    nonzero_counts = [count for count in class_counts.values() if count > 0]
    minority = min(nonzero_counts) if nonzero_counts else 0
    majority = max(nonzero_counts) if nonzero_counts else 0
    imbalance_ratio = float(majority / max(minority, 1)) if minority else float("inf")
    row_order = _row_order_signal(x, y)
    blueprint = _split_blueprint(
        sample_count=dataset.sample_count,
        class_counts=class_counts,
        minority=minority,
        imbalance_ratio=imbalance_ratio,
        row_order=row_order,
    )
    checks = _checks(
        sample_count=dataset.sample_count,
        class_counts=class_counts,
        minority=minority,
        imbalance_ratio=imbalance_ratio,
        row_order=row_order,
    )
    recommendations = _recommendations(blueprint, checks, row_order, imbalance_ratio)
    summary = _summary(blueprint, checks, recommendations, minority, imbalance_ratio, row_order)
    return {
        "sample_count": int(dataset.sample_count),
        "input_dim": int(dataset.input_dim),
        "class_counts": class_counts,
        "dataset_fingerprint": validation_plan_dataset_fingerprint(x, y),
        "summary": summary,
        "split_blueprint": blueprint,
        "row_order": row_order,
        "checks": checks,
        "recommendations": recommendations,
    }


def format_validation_plan_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Validation plan: "
        f"strategy={summary.get('recommended_strategy', '-')}, "
        f"risk={summary.get('risk_level', '-')}, "
        f"readiness={float(summary.get('readiness_score', 0.0)):.1f}/100, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def validation_plan_dataset_fingerprint(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _sequence_to_lists(value: Sequence[Any] | np.ndarray) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _row_order_signal(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    sample_count = int(x.shape[0])
    if sample_count < 12:
        return {
            "available": False,
            "reason": "at least 12 rows are needed for a reference/current row-order screen",
            "prevalence_delta": 0.0,
            "max_standardized_mean_shift": 0.0,
            "top_shift_feature": None,
            "row_order_risk": False,
        }
    split = sample_count // 2
    first_x = x[:split]
    second_x = x[split:]
    first_y = y[:split]
    second_y = y[split:]
    prevalence_delta = abs(float(np.mean(second_y)) - float(np.mean(first_y)))
    scale = np.std(x, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    shifts = np.abs(np.mean(second_x, axis=0) - np.mean(first_x, axis=0)) / scale
    top_feature = int(np.argmax(shifts)) if shifts.size else None
    max_shift = float(np.max(shifts)) if shifts.size else 0.0
    row_order_risk = prevalence_delta >= 0.25 or max_shift >= 0.80
    return {
        "available": True,
        "reference_rows": int(split),
        "current_rows": int(sample_count - split),
        "reference_prevalence": float(np.mean(first_y)),
        "current_prevalence": float(np.mean(second_y)),
        "prevalence_delta": float(prevalence_delta),
        "max_standardized_mean_shift": max_shift,
        "top_shift_feature": top_feature,
        "row_order_risk": bool(row_order_risk),
    }


def _split_blueprint(
    *,
    sample_count: int,
    class_counts: dict[str, int],
    minority: int,
    imbalance_ratio: float,
    row_order: dict[str, Any],
) -> dict[str, Any]:
    two_classes = all(count > 0 for count in class_counts.values())
    row_order_risk = bool(row_order.get("row_order_risk"))
    if not two_classes or sample_count < 12 or minority < 3:
        strategy = "collect_more_labels"
        train_ratio = None
        validation_fraction = None
        kfold_splits = None
        shuffle = True
        stratify = two_classes
    elif row_order_risk and sample_count >= 24:
        strategy = "chronological_holdout"
        train_ratio = 0.70
        validation_fraction = 0.30
        kfold_splits = min(5, minority) if minority >= 3 else None
        shuffle = False
        stratify = False
    elif sample_count < 60 or minority < 8:
        strategy = "stratified_kfold"
        train_ratio = None
        validation_fraction = None
        kfold_splits = min(5, minority)
        shuffle = True
        stratify = True
    else:
        strategy = "stratified_holdout"
        validation_fraction = 0.20 if sample_count >= 100 and minority >= 12 else 0.25
        train_ratio = 1.0 - validation_fraction
        kfold_splits = min(5, minority)
        shuffle = True
        stratify = True
    notes: list[str] = []
    if imbalance_ratio >= 4.0:
        notes.append("Keep stratification, class-weighted training, and precision/recall metrics in the validation report.")
    if row_order_risk:
        notes.append("Row order appears meaningful; avoid shuffled-only evidence until chronological validation is reviewed.")
    if strategy != "collect_more_labels":
        notes.append("Fit preprocessing and feature selection inside each split, not on the full dataset.")
    return {
        "strategy": strategy,
        "train_ratio": train_ratio,
        "validation_fraction": validation_fraction,
        "kfold_splits": int(kfold_splits) if kfold_splits else None,
        "shuffle": bool(shuffle),
        "stratify": bool(stratify),
        "external_holdout_recommended": strategy != "collect_more_labels",
        "notes": notes,
    }


def _checks(
    *,
    sample_count: int,
    class_counts: dict[str, int],
    minority: int,
    imbalance_ratio: float,
    row_order: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, evidence: str, action: str) -> None:
        checks.append({"name": name, "status": status, "evidence": evidence, "action": action})

    two_classes = all(count > 0 for count in class_counts.values())
    add(
        "two_classes",
        "pass" if two_classes else "fail",
        f"class_counts={class_counts}",
        "Collect examples from both labels before splitting." if not two_classes else "Use stratified splitting.",
    )
    add(
        "minimum_rows",
        "pass" if sample_count >= 12 else "fail",
        f"sample_count={sample_count}",
        "Collect at least 12 labeled rows before trusting validation metrics." if sample_count < 12 else "Proceed.",
    )
    add(
        "minority_class_per_split",
        "pass" if minority >= 3 else "fail",
        f"minority_count={minority}",
        "Collect at least 3 rows in the minority class for split validation." if minority < 3 else "Proceed.",
    )
    add(
        "imbalance",
        "review" if imbalance_ratio >= 4.0 else "pass",
        f"imbalance_ratio={imbalance_ratio:.3f}" if np.isfinite(imbalance_ratio) else "imbalance_ratio=inf",
        "Use class weights, threshold tradeoffs, and precision/recall reporting." if imbalance_ratio >= 4.0 else "Proceed.",
    )
    add(
        "row_order",
        "review" if row_order.get("row_order_risk") else "pass",
        (
            f"prevalence_delta={float(row_order.get('prevalence_delta', 0.0)):.3f}, "
            f"max_shift={float(row_order.get('max_standardized_mean_shift', 0.0)):.3f}"
        ),
        "Run chronological holdout or preserve row order during validation." if row_order.get("row_order_risk") else "Proceed.",
    )
    return checks


def _recommendations(
    blueprint: dict[str, Any],
    checks: list[dict[str, Any]],
    row_order: dict[str, Any],
    imbalance_ratio: float,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []

    def add(rank: int, priority: str, category: str, title: str, action: str) -> None:
        recommendations.append(
            {"rank": int(rank), "priority": priority, "category": category, "title": title, "action": action}
        )

    strategy = str(blueprint.get("strategy", "collect_more_labels"))
    failed = {check["name"] for check in checks if check["status"] == "fail"}
    if failed:
        add(1, "high", "data", "Collect more split-safe labels", "Add rows until both classes have at least 3 examples.")
    elif strategy == "chronological_holdout":
        add(1, "high", "validation", "Use chronological validation first", "Train on earlier rows and evaluate on later rows before shuffled CV.")
    elif strategy == "stratified_kfold":
        add(1, "high", "validation", "Use stratified cross-validation", f"Use {blueprint.get('kfold_splits')} folds and report mean/spread.")
    else:
        add(1, "high", "validation", "Use stratified holdout", f"Reserve {blueprint.get('validation_fraction')} of rows for validation.")
    if imbalance_ratio >= 4.0:
        add(2, "medium", "metrics", "Report imbalance-aware metrics", "Track precision, recall, balanced accuracy, PR-AUC, and threshold tradeoffs.")
    if row_order.get("row_order_risk"):
        add(3, "medium", "drift", "Run ordered-data diagnostics", "Run Population drift, Adversarial validation, and Chronological holdout.")
    if strategy != "collect_more_labels":
        add(4, "medium", "holdout", "Keep a final external holdout", "Score a separate labeled CSV once model selection is finished.")
    return recommendations


def _summary(
    blueprint: dict[str, Any],
    checks: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    minority: int,
    imbalance_ratio: float,
    row_order: dict[str, Any],
) -> dict[str, Any]:
    penalty = 0.0
    for check in checks:
        if check["status"] == "fail":
            penalty += 18.0
        elif check["status"] == "review":
            penalty += 8.0
    if bool(row_order.get("row_order_risk")):
        penalty += 4.0
    if imbalance_ratio >= 8.0:
        penalty += 6.0
    readiness = max(0.0, 100.0 - penalty)
    if blueprint.get("strategy") == "collect_more_labels" or readiness < 60.0:
        risk = "high"
    elif readiness < 82.0 or bool(row_order.get("row_order_risk")):
        risk = "medium"
    else:
        risk = "low"
    top = recommendations[0]["action"] if recommendations else None
    warning = None
    if blueprint.get("strategy") == "collect_more_labels":
        warning = "Validation evidence is not split-safe yet."
    elif row_order.get("row_order_risk"):
        warning = "Row-order drift may make shuffled-only validation optimistic."
    return {
        "recommended_strategy": blueprint.get("strategy"),
        "risk_level": risk,
        "readiness_score": round(float(readiness), 1),
        "minority_class_count": int(minority),
        "imbalance_ratio": float(imbalance_ratio) if np.isfinite(imbalance_ratio) else None,
        "kfold_splits": blueprint.get("kfold_splits"),
        "validation_fraction": blueprint.get("validation_fraction"),
        "row_order_risk": bool(row_order.get("row_order_risk")),
        "recommended_next_step": top,
        "warning": warning,
    }
