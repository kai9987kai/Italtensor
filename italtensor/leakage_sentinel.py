from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


EPSILON = 1e-12


def run_leakage_sentinel(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    max_features: int = 12,
    max_mapping_values: int = 24,
    min_value_count: int = 2,
) -> dict[str, Any]:
    """Scan numeric feature columns for target leakage and proxy-shortcut risk."""
    x, y = _validate_inputs(features, labels)
    max_features = _positive_int(max_features, "max_features")
    max_mapping_values = _positive_int(max_mapping_values, "max_mapping_values")
    min_value_count = _positive_int(min_value_count, "min_value_count")

    rows = [
        _feature_row(
            x[:, index],
            y,
            index,
            max_mapping_values=max_mapping_values,
            min_value_count=min_value_count,
        )
        for index in range(x.shape[1])
    ]
    rows.sort(
        key=lambda item: (
            -float(item["risk_score"]),
            -float(item["label_mapping_balanced_accuracy"] or 0.0),
            -float(item["best_balanced_accuracy"]),
            int(item["feature_index"]),
        )
    )
    summary = _summary(rows)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dataset_fingerprint": leakage_sentinel_dataset_fingerprint(x, y),
        "max_mapping_values": int(max_mapping_values),
        "min_value_count": int(min_value_count),
        "features": rows[:max_features],
        "summary": summary,
        "recommendations": _recommendations(summary),
    }


def format_leakage_sentinel_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top = summary.get("top_feature")
    top_text = "-" if top is None else f"x{int(top) + 1}"
    return (
        "Leakage sentinel: "
        f"risk={summary.get('risk_level', '-')}, "
        f"top={top_text}, "
        f"score={float(summary.get('max_risk_score', 0.0)):.4f}, "
        f"high={int(summary.get('high_risk_feature_count', 0))}, "
        f"medium={int(summary.get('medium_risk_feature_count', 0))}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def leakage_sentinel_dataset_fingerprint(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> str:
    x, y = _validate_inputs(features, labels)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(value) for value in x.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    hasher.update(str(tuple(int(value) for value in y.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return hasher.hexdigest()


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Leakage sentinel features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Leakage sentinel features must be a 2D array.")
    if x.shape[0] < 6:
        raise ValueError("Leakage sentinel needs at least six rows.")
    if x.shape[1] == 0:
        raise ValueError("Leakage sentinel needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Leakage sentinel features must be finite numbers.")

    try:
        y_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Leakage sentinel labels must be numeric.") from exc
    if y_values.shape[0] != x.shape[0]:
        raise ValueError("Leakage sentinel feature and label counts do not match.")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("Leakage sentinel labels must be finite numbers.")
    if not np.all(y_values == np.round(y_values)):
        raise ValueError("Leakage sentinel requires integer binary labels 0 or 1.")
    y = y_values.astype(np.int32)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Leakage sentinel requires binary labels 0 or 1.")
    for class_value in (0, 1):
        if int(np.sum(y == class_value)) < 2:
            raise ValueError("Leakage sentinel needs at least two rows per class.")
    return x, y


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Leakage sentinel {name} must be positive.")
    return parsed


def _feature_row(
    values: np.ndarray,
    labels: np.ndarray,
    index: int,
    *,
    max_mapping_values: int,
    min_value_count: int,
) -> dict[str, Any]:
    raw_auc = _mann_whitney_auc(values[labels == 0], values[labels == 1])
    auc = max(raw_auc, 1.0 - raw_auc)
    direction = "positive_high" if raw_auc >= 0.5 else "negative_high"
    threshold = _best_threshold(values, labels, direction)
    mapping = _mapping_metrics(values, labels, max_mapping_values=max_mapping_values, min_value_count=min_value_count)
    unique_count = int(np.unique(values).shape[0])
    unique_ratio = float(unique_count / max(1, values.shape[0]))
    low_cardinality = unique_count <= max(4, min(max_mapping_values, int(np.ceil(values.shape[0] * 0.20))))
    risk_score = _risk_score(
        auc=float(auc),
        threshold_balanced_accuracy=float(threshold["balanced_accuracy"]),
        mapping_balanced_accuracy=mapping["balanced_accuracy"],
        pure_coverage=float(mapping["pure_coverage"]),
        low_cardinality=low_cardinality,
        unique_count=unique_count,
    )
    risk_flags = _risk_flags(
        auc=float(auc),
        threshold_balanced_accuracy=float(threshold["balanced_accuracy"]),
        mapping=mapping,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        low_cardinality=low_cardinality,
        risk_score=risk_score,
    )
    return {
        "feature_index": int(index),
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "low_cardinality": bool(low_cardinality),
        "auc": float(auc),
        "raw_auc": float(raw_auc),
        "direction": direction,
        "best_threshold": float(threshold["threshold"]),
        "best_balanced_accuracy": float(threshold["balanced_accuracy"]),
        "best_accuracy": float(threshold["accuracy"]),
        "label_mapping_accuracy": mapping["accuracy"],
        "label_mapping_balanced_accuracy": mapping["balanced_accuracy"],
        "pure_value_count": int(mapping["pure_value_count"]),
        "mixed_value_count": int(mapping["mixed_value_count"]),
        "pure_coverage": float(mapping["pure_coverage"]),
        "majority_mapping_coverage": float(mapping["coverage"]),
        "top_values": mapping["top_values"],
        "risk_score": float(risk_score),
        "risk_level": _risk_level(risk_score, risk_flags),
        "risk_flags": risk_flags,
    }


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
        predicted = (values >= threshold).astype(np.int32) if direction == "positive_high" else (values <= threshold).astype(np.int32)
        metrics = _classification_metrics(labels, predicted)
        if (metrics["balanced_accuracy"], metrics["accuracy"]) > (best["balanced_accuracy"], best["accuracy"]):
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
            }
    return best


def _mapping_metrics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    max_mapping_values: int,
    min_value_count: int,
) -> dict[str, Any]:
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if unique.shape[0] > max_mapping_values:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "pure_value_count": 0,
            "mixed_value_count": 0,
            "pure_coverage": 0.0,
            "coverage": 0.0,
            "top_values": [],
        }

    predicted = np.zeros(labels.shape[0], dtype=np.int32)
    pure_count = 0
    mixed_count = 0
    pure_rows = 0
    covered_rows = 0
    value_rows: list[dict[str, Any]] = []
    for position, value in enumerate(unique):
        mask = inverse == position
        count = int(np.sum(mask))
        if count < min_value_count:
            majority = 1 if float(np.mean(labels[mask])) >= 0.5 else 0
            predicted[mask] = majority
            continue
        positives = int(np.sum(labels[mask] == 1))
        negatives = int(count - positives)
        positive_rate = float(positives / max(count, 1))
        majority = 1 if positives >= negatives else 0
        predicted[mask] = majority
        covered_rows += count
        if positives == 0 or negatives == 0:
            pure_count += 1
            pure_rows += count
        else:
            mixed_count += 1
        value_rows.append(
            {
                "value": float(value),
                "count": count,
                "positive_rate": positive_rate,
                "majority_label": int(majority),
                "pure": bool(positives == 0 or negatives == 0),
            }
        )

    metrics = _classification_metrics(labels, predicted)
    value_rows.sort(key=lambda item: (-int(item["count"]), -abs(float(item["positive_rate"]) - 0.5), float(item["value"])))
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "pure_value_count": pure_count,
        "mixed_value_count": mixed_count,
        "pure_coverage": float(pure_rows / max(1, labels.shape[0])),
        "coverage": float(covered_rows / max(1, labels.shape[0])),
        "top_values": value_rows[:8],
    }


def _classification_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    recall_pos = tp / max(tp + fn, 1)
    recall_neg = tn / max(tn + fp, 1)
    return {
        "accuracy": float((tp + tn) / max(1, labels.shape[0])),
        "balanced_accuracy": float(0.5 * (recall_pos + recall_neg)),
    }


def _risk_score(
    *,
    auc: float,
    threshold_balanced_accuracy: float,
    mapping_balanced_accuracy: float | None,
    pure_coverage: float,
    low_cardinality: bool,
    unique_count: int,
) -> float:
    threshold_strength = max(0.0, (max(auc, threshold_balanced_accuracy) - 0.5) / 0.5)
    mapping_strength = 0.0 if mapping_balanced_accuracy is None else max(0.0, (mapping_balanced_accuracy - 0.5) / 0.5)
    score = 0.58 * threshold_strength + 0.32 * mapping_strength + 0.10 * pure_coverage
    if low_cardinality and mapping_balanced_accuracy is not None:
        score += 0.08 * mapping_strength
    if unique_count <= 2 and mapping_balanced_accuracy is not None and mapping_balanced_accuracy >= 0.98:
        score = max(score, 0.99)
    return float(np.clip(score, 0.0, 1.0))


def _risk_flags(
    *,
    auc: float,
    threshold_balanced_accuracy: float,
    mapping: dict[str, Any],
    unique_count: int,
    unique_ratio: float,
    low_cardinality: bool,
    risk_score: float,
) -> list[str]:
    flags: list[str] = []
    mapping_bal = mapping["balanced_accuracy"]
    pure_coverage = float(mapping["pure_coverage"])
    if unique_count <= 2 and mapping_bal is not None and mapping_bal >= 0.98:
        flags.append("direct_label_copy_candidate")
    if auc >= 0.995 and threshold_balanced_accuracy >= 0.98:
        flags.append("near_perfect_single_feature")
    elif auc >= 0.92 or threshold_balanced_accuracy >= 0.90:
        flags.append("strong_single_feature_proxy")
    if low_cardinality and mapping_bal is not None and mapping_bal >= 0.90:
        flags.append("low_cardinality_label_mapping")
    if pure_coverage >= 0.80 and int(mapping["pure_value_count"]) > 0:
        flags.append("mostly_pure_feature_values")
    if unique_ratio >= 0.80 and risk_score >= 0.82:
        flags.append("high_cardinality_proxy")
    if not flags and risk_score >= 0.55:
        flags.append("review_proxy_risk")
    return flags


def _risk_level(risk_score: float, flags: list[str]) -> str:
    high_markers = {"direct_label_copy_candidate", "near_perfect_single_feature", "low_cardinality_label_mapping"}
    if risk_score >= 0.85 or high_markers.intersection(flags):
        return "high"
    if risk_score >= 0.55 or "strong_single_feature_proxy" in flags:
        return "medium"
    return "low"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = rows[0] if rows else None
    high_count = int(sum(item["risk_level"] == "high" for item in rows))
    medium_count = int(sum(item["risk_level"] == "medium" for item in rows))
    direct_count = int(sum("direct_label_copy_candidate" in item["risk_flags"] for item in rows))
    low_cardinality_mapping_count = int(sum("low_cardinality_label_mapping" in item["risk_flags"] for item in rows))
    risk_level = "high" if high_count else ("medium" if medium_count else "low")
    recommendation = _next_step(risk_level, top)
    return {
        "risk_level": risk_level,
        "top_feature": None if top is None else int(top["feature_index"]),
        "max_risk_score": 0.0 if top is None else float(top["risk_score"]),
        "top_feature_auc": 0.0 if top is None else float(top["auc"]),
        "top_feature_balanced_accuracy": 0.0 if top is None else float(top["best_balanced_accuracy"]),
        "top_label_mapping_balanced_accuracy": None if top is None else top["label_mapping_balanced_accuracy"],
        "high_risk_feature_count": high_count,
        "medium_risk_feature_count": medium_count,
        "direct_label_copy_candidate_count": direct_count,
        "low_cardinality_label_mapping_count": low_cardinality_mapping_count,
        "recommendation": recommendation,
        "warning": recommendation if risk_level != "low" else None,
    }


def _next_step(risk_level: str, top: dict[str, Any] | None) -> str:
    if top is None:
        return "Load a dataset with at least one numeric feature before running leakage checks."
    feature = f"x{int(top['feature_index']) + 1}"
    if risk_level == "high":
        return f"Quarantine {feature} until you can prove it is available at prediction time and not derived from the label."
    if risk_level == "medium":
        return f"Review {feature} as a possible proxy or shortcut before model selection."
    return "Keep this leakage-sentinel evidence with the dataset report."


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    risk = str(summary.get("risk_level", "low"))
    priority = "high" if risk == "high" else ("medium" if risk == "medium" else "low")
    category = "leakage" if risk != "low" else "evidence"
    title = (
        "Quarantine possible target leakage"
        if risk == "high"
        else ("Review possible proxy shortcut" if risk == "medium" else "Retain leakage sentinel evidence")
    )
    return [
        {
            "rank": 1,
            "priority_score": 92.0 if risk == "high" else (66.0 if risk == "medium" else 22.0),
            "priority": priority,
            "category": category,
            "title": title,
            "action": str(summary.get("recommendation") or "Review leakage sentinel output."),
        }
    ]
