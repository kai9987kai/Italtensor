"""Model-free planning for the next useful labels to collect."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def run_data_acquisition_planner(
    features: Any,
    labels: Any,
    *,
    min_class_count: int = 20,
    target_minority_fraction: float = 0.35,
    max_candidates: int = 8,
) -> dict[str, Any]:
    """Build a deterministic label-acquisition plan from the loaded numeric dataset."""
    x, y = _validate_inputs(features, labels)
    n_samples, input_dim = x.shape
    class_counts = {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}
    target_minority_fraction = min(max(float(target_minority_fraction), 0.05), 0.50)
    max_candidates = max(1, int(max_candidates))

    robust_z = _robust_z_scores(x)
    boundary_scores = _boundary_scores(x, y)
    tail_scores = np.max(np.abs(robust_z), axis=1)
    feature_targets = _feature_targets(x, y, robust_z)
    row_candidates = _row_candidates(y, boundary_scores, tail_scores, max_candidates=max_candidates)
    recommendations = _recommendations(
        n_samples=n_samples,
        input_dim=input_dim,
        class_counts=class_counts,
        target_minority_fraction=target_minority_fraction,
        min_class_count=max(1, int(min_class_count)),
        row_candidates=row_candidates,
        feature_targets=feature_targets,
    )
    summary = _summary(
        n_samples=n_samples,
        class_counts=class_counts,
        recommendations=recommendations,
        row_candidates=row_candidates,
        feature_targets=feature_targets,
    )
    return {
        "sample_count": int(n_samples),
        "input_dim": int(input_dim),
        "class_counts": class_counts,
        "dataset_fingerprint": data_acquisition_dataset_fingerprint(x, y),
        "settings": {
            "min_class_count": int(min_class_count),
            "target_minority_fraction": float(target_minority_fraction),
            "max_candidates": int(max_candidates),
        },
        "summary": summary,
        "recommendations": recommendations,
        "row_candidates": row_candidates,
        "feature_targets": feature_targets,
    }


def format_data_acquisition_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Data acquisition plan: "
        f"verdict={summary.get('verdict', '-')}, "
        f"priority={summary.get('priority', '-')}, "
        f"budget={int(summary.get('recommended_label_budget', 0) or 0)}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def data_acquisition_dataset_fingerprint(features: Any, labels: Any) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_inputs(features: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Data acquisition planning needs a 2D feature matrix.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature and label counts do not match.")
    if x.shape[0] < 2:
        raise ValueError("Data acquisition planning needs at least two labeled rows.")
    if x.shape[1] < 1:
        raise ValueError("Data acquisition planning needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must be finite numbers.")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("Labels must be binary values 0 or 1.")
    return x, y


def _robust_z_scores(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    scale = np.where(mad > 1e-8, 1.4826 * mad, np.std(x, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (x - median) / scale


def _standardized_features(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (x - mean) / std


def _boundary_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if np.unique(y).size < 2:
        return np.zeros(y.shape[0], dtype=np.float32)
    z = _standardized_features(x)
    center_0 = np.mean(z[y == 0], axis=0)
    center_1 = np.mean(z[y == 1], axis=0)
    dist_0 = np.linalg.norm(z - center_0, axis=1)
    dist_1 = np.linalg.norm(z - center_1, axis=1)
    denom = dist_0 + dist_1 + 1e-8
    scores = 1.0 - np.abs(dist_0 - dist_1) / denom
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def _feature_targets(x: np.ndarray, y: np.ndarray, robust_z: np.ndarray) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    two_classes = np.unique(y).size == 2
    for index in range(x.shape[1]):
        values = x[:, index]
        tail_rate = float(np.mean(np.abs(robust_z[:, index]) >= 3.5))
        missing_tail_labels = int(np.sum(np.abs(robust_z[:, index]) >= 3.5))
        if two_classes:
            pooled = float(np.std(values))
            pooled = pooled if pooled > 1e-8 else 1.0
            class_gap = float(abs(np.mean(values[y == 1]) - np.mean(values[y == 0])) / pooled)
        else:
            class_gap = 0.0
        priority_score = float(40.0 * tail_rate + max(0.0, 0.35 - min(class_gap, 0.35)) * 20.0)
        if tail_rate >= 0.08:
            priority = "high"
            action = "Collect or review more labels from this feature's tail range."
        elif class_gap < 0.15:
            priority = "medium"
            action = "Collect examples that make this weak feature informative or confirm it is noise."
        else:
            priority = "low"
            action = "Coverage looks adequate for this feature."
        targets.append(
            {
                "feature_index": int(index),
                "feature_name": f"x{index + 1}",
                "class_gap": class_gap,
                "tail_rate": tail_rate,
                "tail_row_count": missing_tail_labels,
                "priority": priority,
                "priority_score": priority_score,
                "action": action,
            }
        )
    targets.sort(key=lambda item: (-float(item["priority_score"]), int(item["feature_index"])))
    for rank, item in enumerate(targets, start=1):
        item["rank"] = rank
    return targets[: min(8, len(targets))]


def _row_candidates(
    y: np.ndarray,
    boundary_scores: np.ndarray,
    tail_scores: np.ndarray,
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, score in enumerate(boundary_scores):
        if float(score) >= 0.70:
            candidates.append(
                {
                    "row_index": int(index),
                    "label": int(y[index]),
                    "candidate_type": "boundary",
                    "score": float(score),
                    "reason": "Near both class centroids; useful for boundary labeling or review.",
                }
            )
    for index, score in enumerate(tail_scores):
        if float(score) >= 3.5:
            candidates.append(
                {
                    "row_index": int(index),
                    "label": int(y[index]),
                    "candidate_type": "tail_coverage",
                    "score": float(score / 5.0),
                    "reason": "Feature tail row; useful for coverage and schema review.",
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), item["candidate_type"], int(item["row_index"])))
    deduped: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    for item in candidates:
        row_index = int(item["row_index"])
        if row_index in seen_rows:
            continue
        seen_rows.add(row_index)
        row = dict(item)
        row["rank"] = len(deduped) + 1
        deduped.append(row)
        if len(deduped) >= max_candidates:
            break
    return deduped


def _recommendations(
    *,
    n_samples: int,
    input_dim: int,
    class_counts: dict[str, int],
    target_minority_fraction: float,
    min_class_count: int,
    row_candidates: list[dict[str, Any]],
    feature_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def add(score: float, priority: str, category: str, title: str, reason: str, action: str, budget: int) -> None:
        recs.append(
            {
                "priority": priority,
                "priority_score": float(score),
                "category": category,
                "title": title,
                "reason": reason,
                "action": action,
                "suggested_label_budget": int(max(0, budget)),
            }
        )

    seed_goal = max(40, input_dim * 10)
    if n_samples < seed_goal:
        add(
            92.0,
            "high",
            "volume",
            "Collect a stronger seed dataset",
            f"{n_samples} labeled rows are loaded; a practical seed target is {seed_goal}.",
            "Collect broad, ordinary examples before trusting fine-grained diagnostics.",
            seed_goal - n_samples,
        )

    count_0 = class_counts["0"]
    count_1 = class_counts["1"]
    if count_0 == 0 or count_1 == 0:
        missing = 0 if count_0 == 0 else 1
        add(
            100.0,
            "high",
            "class_balance",
            f"Collect class {missing} examples",
            f"Class counts are {class_counts}; both classes are required for validation.",
            f"Prioritize verified class {missing} rows before model selection.",
            min_class_count,
        )
    else:
        minority_label = 0 if count_0 <= count_1 else 1
        minority = min(count_0, count_1)
        needed_for_floor = max(0, min_class_count - minority)
        needed_for_fraction = int(np.ceil(max(0.0, target_minority_fraction * n_samples - minority) / (1.0 - target_minority_fraction)))
        needed = max(needed_for_floor, needed_for_fraction)
        if needed > 0:
            add(
                88.0,
                "high" if minority < min_class_count else "medium",
                "class_balance",
                f"Collect more class {minority_label} labels",
                f"Class counts are {class_counts}; class {minority_label} is the limiting class.",
                f"Collect at least {needed} verified class {minority_label} rows or use reviewed batch import.",
                needed,
            )

    boundary_rows = [item for item in row_candidates if item["candidate_type"] == "boundary"]
    if boundary_rows:
        add(
            70.0,
            "medium",
            "boundary",
            "Review or label boundary-like rows",
            f"{len(boundary_rows)} loaded row(s) sit near both class centroids.",
            "Use active review, canary examples, or new CSV rows around these boundary regions.",
            min(max(8, len(boundary_rows) * 2), 40),
        )

    tail_rows = [item for item in row_candidates if item["candidate_type"] == "tail_coverage"]
    if tail_rows:
        add(
            64.0,
            "medium",
            "coverage",
            "Add coverage for feature-tail rows",
            f"{len(tail_rows)} loaded row(s) are feature-tail coverage candidates.",
            "Confirm whether these tails are valid cases, import mistakes, or future rows needing labels.",
            min(max(6, len(tail_rows) * 2), 32),
        )

    weak_features = [item for item in feature_targets if item["priority"] in {"high", "medium"}]
    if weak_features:
        top = weak_features[0]
        add(
            52.0,
            "low",
            "feature_coverage",
            f"Improve evidence for {top['feature_name']}",
            f"{top['feature_name']} has tail_rate={top['tail_rate']:.3f}, class_gap={top['class_gap']:.3f}.",
            str(top["action"]),
            8,
        )

    if not recs:
        add(
            20.0,
            "low",
            "ready",
            "Proceed to validation and model search",
            "Class balance, boundary candidates, and feature coverage look adequate for the current dataset.",
            "Run Validation plan, auto experiments, and an external holdout before promotion.",
            0,
        )

    recs.sort(key=lambda item: (-float(item["priority_score"]), item["category"], item["title"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs


def _summary(
    *,
    n_samples: int,
    class_counts: dict[str, int],
    recommendations: list[dict[str, Any]],
    row_candidates: list[dict[str, Any]],
    feature_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    top = recommendations[0] if recommendations else {}
    high_count = sum(1 for item in recommendations if item.get("priority") == "high")
    medium_count = sum(1 for item in recommendations if item.get("priority") == "medium")
    boundary_count = sum(1 for item in row_candidates if item.get("candidate_type") == "boundary")
    tail_count = sum(1 for item in row_candidates if item.get("candidate_type") == "tail_coverage")
    feature_review_count = sum(1 for item in feature_targets if item.get("priority") in {"high", "medium"})
    readiness_score = 100.0
    readiness_score -= 18.0 * high_count
    readiness_score -= 7.0 * medium_count
    readiness_score -= min(20.0, max(0.0, 40.0 - n_samples) * 0.5)
    readiness_score = max(0.0, min(100.0, readiness_score))
    if high_count:
        verdict = "collect_before_model_selection"
        priority = "high"
    elif medium_count:
        verdict = "targeted_collection_recommended"
        priority = "medium"
    else:
        verdict = "coverage_ready_for_next_experiment"
        priority = "low"
    return {
        "verdict": verdict,
        "priority": priority,
        "readiness_score": round(float(readiness_score), 1),
        "recommended_label_budget": int(max((item.get("suggested_label_budget", 0) for item in recommendations), default=0)),
        "recommendation_count": int(len(recommendations)),
        "high_priority_count": int(high_count),
        "medium_priority_count": int(medium_count),
        "boundary_candidate_count": int(boundary_count),
        "tail_candidate_count": int(tail_count),
        "feature_review_count": int(feature_review_count),
        "class_counts": dict(class_counts),
        "recommended_next_step": top.get("action"),
    }
