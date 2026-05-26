from __future__ import annotations

from typing import Any, Sequence

import numpy as np


EPSILON = 1e-12


def run_feature_separability_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    max_features: int = 12,
) -> dict[str, Any]:
    """Score one-feature class separation, leakage-like strength, and redundancy."""
    x, y = _validate_inputs(features, labels)
    max_features = _validate_positive_int(max_features, "max_features")

    feature_rows = [_feature_row(x[:, index], y, index) for index in range(x.shape[1])]
    feature_rows.sort(
        key=lambda row: (
            -float(row["separability_score"]),
            -float(row["best_balanced_accuracy"]),
            -float(row["auc"]),
            int(row["feature_index"]),
        )
    )
    redundant_pairs = _redundant_pairs(x, feature_rows)
    summary = _summary(feature_rows, redundant_pairs)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "class_counts": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "summary": summary,
        "features": feature_rows[:max_features],
        "redundant_pairs": redundant_pairs[:max_features],
    }


def format_feature_separability_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_feature = summary.get("top_feature")
    top_text = "-" if top_feature is None else f"x{int(top_feature) + 1}"
    return (
        "Feature separability: "
        f"top={top_text}, "
        f"top_auc={float(summary.get('top_auc', 0.0)):.4f}, "
        f"top_bal_acc={float(summary.get('top_balanced_accuracy', 0.0)):.4f}, "
        f"near_perfect={int(summary.get('near_perfect_feature_count', 0))}, "
        f"weak={int(summary.get('weak_feature_count', 0))}, "
        f"redundant_pairs={int(summary.get('redundant_pair_count', 0))}"
    )


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature separability features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Feature separability features must be a 2D array.")
    if x.shape[0] < 6:
        raise ValueError("Feature separability needs at least six rows.")
    if x.shape[1] == 0:
        raise ValueError("Feature separability needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Feature separability features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature separability labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Feature separability feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Feature separability labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Feature separability requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Feature separability requires binary labels 0 or 1.")
    for class_value in (0, 1):
        if int(np.sum(y == class_value)) < 2:
            raise ValueError("Feature separability needs at least two rows per class.")
    return x, y


def _validate_positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Feature separability {name} must be positive.")
    return parsed


def _feature_row(values: np.ndarray, labels: np.ndarray, index: int) -> dict[str, Any]:
    negative = values[labels == 0]
    positive = values[labels == 1]
    mean_0 = float(np.mean(negative))
    mean_1 = float(np.mean(positive))
    std_0 = float(np.std(negative))
    std_1 = float(np.std(positive))
    pooled = float(np.sqrt((std_0 * std_0 + std_1 * std_1) / 2.0))
    effect = abs(mean_1 - mean_0) / max(pooled, EPSILON)
    raw_auc = _mann_whitney_auc(negative, positive)
    auc = max(raw_auc, 1.0 - raw_auc)
    direction = "positive_high" if raw_auc >= 0.5 else "negative_high"
    threshold = _best_threshold(values, labels, direction)
    separation_score = float(np.clip(0.50 * ((auc - 0.5) / 0.5) + 0.30 * (threshold["balanced_accuracy"] - 0.5) / 0.5 + 0.20 * min(effect / 3.0, 1.0), 0.0, 1.0))
    unique_ratio = float(np.unique(values).shape[0] / values.shape[0])
    row = {
        "feature_index": int(index),
        "mean_0": mean_0,
        "mean_1": mean_1,
        "std_0": std_0,
        "std_1": std_1,
        "standardized_mean_difference": float(effect),
        "auc": float(auc),
        "raw_auc": float(raw_auc),
        "direction": direction,
        "best_threshold": float(threshold["threshold"]),
        "best_balanced_accuracy": float(threshold["balanced_accuracy"]),
        "best_accuracy": float(threshold["accuracy"]),
        "separability_score": separation_score,
        "unique_ratio": unique_ratio,
    }
    row["risk_flags"] = _risk_flags(row)
    return row


def _mann_whitney_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    wins = 0.0
    total = float(negative.shape[0] * positive.shape[0])
    for pos_value in positive:
        wins += float(np.sum(pos_value > negative))
        wins += 0.5 * float(np.sum(pos_value == negative))
    return float(wins / max(total, 1.0))


def _best_threshold(values: np.ndarray, labels: np.ndarray, direction: str) -> dict[str, float]:
    ordered = np.unique(values)
    if ordered.shape[0] == 1:
        thresholds = ordered
    else:
        thresholds = (ordered[:-1] + ordered[1:]) / 2.0
        thresholds = np.concatenate(([ordered[0] - EPSILON], thresholds, [ordered[-1] + EPSILON]))
    best = {"threshold": float(thresholds[0]), "balanced_accuracy": -1.0, "accuracy": -1.0}
    for threshold in thresholds:
        if direction == "positive_high":
            predicted = (values >= threshold).astype(np.int32)
        else:
            predicted = (values <= threshold).astype(np.int32)
        tp = int(np.sum((labels == 1) & (predicted == 1)))
        tn = int(np.sum((labels == 0) & (predicted == 0)))
        fp = int(np.sum((labels == 0) & (predicted == 1)))
        fn = int(np.sum((labels == 1) & (predicted == 0)))
        recall_pos = tp / max(tp + fn, 1)
        recall_neg = tn / max(tn + fp, 1)
        balanced_accuracy = 0.5 * (recall_pos + recall_neg)
        accuracy = (tp + tn) / labels.shape[0]
        if (balanced_accuracy, accuracy) > (best["balanced_accuracy"], best["accuracy"]):
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced_accuracy),
                "accuracy": float(accuracy),
            }
    return best


def _risk_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if float(row["best_balanced_accuracy"]) >= 0.98 and float(row["auc"]) >= 0.99:
        flags.append("near_perfect_single_feature")
    elif float(row["auc"]) >= 0.85:
        flags.append("strong_single_feature")
    if float(row["auc"]) <= 0.58 and float(row["best_balanced_accuracy"]) <= 0.62:
        flags.append("weak_single_feature")
    if float(row["unique_ratio"]) <= 0.10 and float(row["best_balanced_accuracy"]) >= 0.90:
        flags.append("low_cardinality_shortcut")
    return flags


def _redundant_pairs(x: np.ndarray, feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_index = {int(row["feature_index"]): row for row in feature_rows}
    pairs: list[dict[str, Any]] = []
    if x.shape[1] < 2:
        return pairs
    corr = np.corrcoef(x, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    for left in range(x.shape[1]):
        for right in range(left + 1, x.shape[1]):
            value = float(corr[left, right]) if np.isfinite(corr[left, right]) else 0.0
            abs_value = abs(value)
            if abs_value < 0.92:
                continue
            strength = min(
                float(rows_by_index[left]["separability_score"]),
                float(rows_by_index[right]["separability_score"]),
            )
            pairs.append(
                {
                    "left_feature_index": int(left),
                    "right_feature_index": int(right),
                    "correlation": value,
                    "abs_correlation": abs_value,
                    "min_separability_score": strength,
                    "risk_flags": ["redundant_strong_features"] if strength >= 0.70 else ["redundant_features"],
                }
            )
    pairs.sort(key=lambda row: (-float(row["abs_correlation"]), -float(row["min_separability_score"])))
    return pairs


def _summary(feature_rows: list[dict[str, Any]], redundant_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    top = feature_rows[0] if feature_rows else None
    near_perfect = int(sum("near_perfect_single_feature" in row["risk_flags"] for row in feature_rows))
    weak = int(sum("weak_single_feature" in row["risk_flags"] for row in feature_rows))
    strong = int(sum(float(row["auc"]) >= 0.85 for row in feature_rows))
    warnings: list[str] = []
    if near_perfect:
        warnings.append(f"{near_perfect} near-perfect single feature(s) may indicate leakage or shortcut risk")
    if weak == len(feature_rows) and feature_rows:
        warnings.append("all features look weak in one-dimensional scans")
    if redundant_pairs:
        warnings.append(f"{len(redundant_pairs)} highly correlated feature pair(s)")
    return {
        "top_feature": None if top is None else int(top["feature_index"]),
        "top_auc": 0.0 if top is None else float(top["auc"]),
        "top_balanced_accuracy": 0.0 if top is None else float(top["best_balanced_accuracy"]),
        "top_separability_score": 0.0 if top is None else float(top["separability_score"]),
        "mean_auc": float(np.mean([row["auc"] for row in feature_rows])) if feature_rows else 0.0,
        "strong_feature_count": strong,
        "weak_feature_count": weak,
        "near_perfect_feature_count": near_perfect,
        "redundant_pair_count": int(len(redundant_pairs)),
        "warning": "; ".join(warnings) if warnings else None,
    }
