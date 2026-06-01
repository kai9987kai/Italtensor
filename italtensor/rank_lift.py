from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DEFAULT_TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)


def run_rank_lift_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    top_fractions: Sequence[float] | None = None,
    deciles: int = 10,
    max_rows: int = 12,
) -> dict[str, Any]:
    """Measure how much positive-label signal concentrates near the top of the score ranking."""
    if model is None:
        raise ValueError("Rank lift needs an active model.")
    x, y = _validate_inputs(features, labels)
    if deciles <= 0:
        raise ValueError("Rank lift deciles must be positive.")
    max_rows = max(1, int(max_rows))
    probabilities = _predict_probabilities(model, x, preprocessor)
    order = np.lexsort((np.arange(probabilities.shape[0]), -probabilities))
    grid = _top_fraction_grid(top_fractions, x.shape[0])
    points = [_rank_point(fraction, order, y, probabilities) for fraction in grid]
    buckets = _decile_table(order, y, probabilities, deciles=int(deciles))
    top_rows = _top_rows(order, y, probabilities, max_rows=max_rows)
    summary = _summary(points, buckets, y, probabilities)
    recommendations = _recommendations(summary)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "positive_count": int(np.sum(y == 1)),
        "top_fractions": [float(value) for value in grid],
        "deciles": int(min(deciles, x.shape[0])),
        "dataset_fingerprint": rank_lift_dataset_fingerprint(x, y),
        "summary": summary,
        "points": points,
        "deciles_table": buckets,
        "top_rows": top_rows,
        "recommendations": recommendations,
    }


def format_rank_lift_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Rank lift: "
        f"verdict={summary.get('verdict', '-')}, "
        f"prevalence={float(summary.get('prevalence', 0.0)):.4f}, "
        f"top10_lift={float(summary.get('top_10_lift', 0.0)):.4f}, "
        f"top20_capture={float(summary.get('top_20_positive_capture', 0.0)):.4f}, "
        f"gains_auc={float(summary.get('normalized_gains_auc', 0.0)):.4f}, "
        f"score_gini={float(summary.get('score_gini', 0.0)):.4f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def rank_lift_dataset_fingerprint(
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
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rank lift features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Rank lift features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Rank lift feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("Rank lift needs at least one sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Rank lift features must be finite numbers.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rank lift labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Rank lift labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Rank lift labels must be binary 0/1.")
    return y.astype(np.int32)


def _predict_probabilities(model: Any, x: np.ndarray, preprocessor: FeatureStandardizer | None) -> np.ndarray:
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Rank lift preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    return np.clip(probabilities, 0.0, 1.0)


def _top_fraction_grid(top_fractions: Sequence[float] | None, sample_count: int) -> np.ndarray:
    raw = np.asarray(DEFAULT_TOP_FRACTIONS if top_fractions is None else list(top_fractions), dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("Rank lift top fractions must contain at least one finite fraction.")
    raw = np.clip(raw, 1.0 / max(1, sample_count), 1.0)
    by_k: dict[int, float] = {}
    for value in np.unique(raw):
        k = min(sample_count, max(1, int(np.ceil(float(value) * sample_count))))
        by_k[k] = float(k / sample_count)
    return np.asarray([by_k[k] for k in sorted(by_k)], dtype=np.float64)


def _rank_point(
    fraction: float,
    order: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    total = int(labels.shape[0])
    positives = int(np.sum(labels == 1))
    k = min(total, max(1, int(np.ceil(float(fraction) * total))))
    selected = order[:k]
    selected_labels = labels[selected]
    tp = int(np.sum(selected_labels == 1))
    fp = int(k - tp)
    precision = float(tp / k) if k else 0.0
    prevalence = float(positives / total) if total else 0.0
    positive_capture = float(tp / positives) if positives else 0.0
    random_capture = float(k / total) if positives else 0.0
    probability_sum = float(np.sum(probabilities))
    selected_probability_sum = float(np.sum(probabilities[selected]))
    return {
        "top_fraction": float(k / total),
        "requested_top_fraction": float(fraction),
        "k": k,
        "threshold_floor": float(probabilities[selected[-1]]) if k else 1.0,
        "true_positive": tp,
        "false_positive": fp,
        "precision_at_k": precision,
        "positive_capture": positive_capture,
        "cumulative_gain": positive_capture,
        "lift": float(precision / prevalence) if prevalence > 0 else 0.0,
        "gain_over_random_capture": float(positive_capture - random_capture),
        "probability_mass_capture": float(selected_probability_sum / probability_sum) if probability_sum > 0 else 0.0,
        "mean_probability": float(np.mean(probabilities[selected])) if k else 0.0,
    }


def _decile_table(order: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, *, deciles: int) -> list[dict[str, Any]]:
    total = int(labels.shape[0])
    positives = int(np.sum(labels == 1))
    prevalence = float(positives / total) if total else 0.0
    buckets: list[dict[str, Any]] = []
    cumulative_positive = 0
    split_indices = np.array_split(order, min(deciles, total))
    rank_start = 1
    for index, bucket_indices in enumerate(split_indices, start=1):
        if bucket_indices.size == 0:
            continue
        bucket_labels = labels[bucket_indices]
        bucket_probabilities = probabilities[bucket_indices]
        positive_count = int(np.sum(bucket_labels == 1))
        cumulative_positive += positive_count
        response_rate = float(positive_count / bucket_indices.size)
        buckets.append(
            {
                "bucket": int(index),
                "rank_start": int(rank_start),
                "rank_end": int(rank_start + bucket_indices.size - 1),
                "count": int(bucket_indices.size),
                "positive_count": positive_count,
                "response_rate": response_rate,
                "lift": float(response_rate / prevalence) if prevalence > 0 else 0.0,
                "cumulative_positive_capture": float(cumulative_positive / positives) if positives else 0.0,
                "probability_min": float(np.min(bucket_probabilities)),
                "probability_max": float(np.max(bucket_probabilities)),
                "probability_mean": float(np.mean(bucket_probabilities)),
            }
        )
        rank_start += int(bucket_indices.size)
    return buckets


def _top_rows(order: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, *, max_rows: int) -> list[dict[str, Any]]:
    rows = []
    for rank, index in enumerate(order[:max_rows], start=1):
        rows.append(
            {
                "rank": int(rank),
                "row_index": int(index),
                "label": int(labels[index]),
                "probability": float(probabilities[index]),
            }
        )
    return rows


def _summary(
    points: list[dict[str, float | int]],
    buckets: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    total = int(labels.shape[0])
    positives = int(np.sum(labels == 1))
    prevalence = float(positives / max(1, total))
    top_10 = _point_at(points, 0.10)
    top_20 = _point_at(points, 0.20)
    top_10_lift = float(top_10.get("lift", 0.0))
    top_20_lift = float(top_20.get("lift", 0.0))
    top_10_capture = float(top_10.get("positive_capture", 0.0))
    top_20_capture = float(top_20.get("positive_capture", 0.0))
    gains_auc = _normalized_gains_auc(labels, probabilities)
    score_gini = _gini(probabilities)
    score_spread = float(np.max(probabilities) - np.min(probabilities)) if probabilities.size else 0.0
    verdict = _verdict(
        positive_count=positives,
        score_spread=score_spread,
        top_10_lift=top_10_lift,
        top_20_lift=top_20_lift,
        top_20_capture=top_20_capture,
        normalized_gains_auc=gains_auc,
    )
    return {
        "verdict": verdict,
        "sample_count": total,
        "positive_count": positives,
        "prevalence": prevalence,
        "top_10_k": int(top_10.get("k", 0)),
        "top_10_lift": top_10_lift,
        "top_10_positive_capture": top_10_capture,
        "top_10_precision": float(top_10.get("precision_at_k", 0.0)),
        "top_20_k": int(top_20.get("k", 0)),
        "top_20_lift": top_20_lift,
        "top_20_positive_capture": top_20_capture,
        "top_20_precision": float(top_20.get("precision_at_k", 0.0)),
        "normalized_gains_auc": gains_auc,
        "score_gini": score_gini,
        "score_spread": score_spread,
        "best_lift": max((float(point.get("lift", 0.0)) for point in points), default=0.0),
        "best_positive_capture_under_30pct": max(
            (float(point.get("positive_capture", 0.0)) for point in points if float(point["top_fraction"]) <= 0.30),
            default=0.0,
        ),
        "first_bucket_lift": float(buckets[0].get("lift", 0.0)) if buckets else 0.0,
        "recommended_next_step": _next_step(verdict),
    }


def _point_at(points: list[dict[str, float | int]], fraction: float) -> dict[str, float | int]:
    eligible = [point for point in points if float(point["top_fraction"]) >= fraction - 1e-12]
    return min(eligible or points, key=lambda point: (float(point["top_fraction"]), int(point["k"])))


def _normalized_gains_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    total = int(labels.shape[0])
    if positives == 0 or total == 0:
        return 0.0
    order = np.lexsort((np.arange(probabilities.shape[0]), -probabilities))
    sorted_labels = labels[order]
    x = np.concatenate([[0.0], np.arange(1, total + 1, dtype=np.float64) / total])
    y = np.concatenate([[0.0], np.cumsum(sorted_labels) / positives])
    observed_area = float(np.trapezoid(y, x))
    prevalence = positives / total
    ideal_area = float(1.0 - prevalence / 2.0)
    denominator = ideal_area - 0.5
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip((observed_area - 0.5) / denominator, 0.0, 1.0))


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(np.clip(values.astype(np.float64), 0.0, None))
    total = float(np.sum(sorted_values))
    if total <= 0.0:
        return 0.0
    n = sorted_values.shape[0]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(np.clip((2.0 * np.sum(ranks * sorted_values) / (n * total)) - ((n + 1.0) / n), 0.0, 1.0))


def _verdict(
    *,
    positive_count: int,
    score_spread: float,
    top_10_lift: float,
    top_20_lift: float,
    top_20_capture: float,
    normalized_gains_auc: float,
) -> str:
    if positive_count == 0:
        return "no_positive_evidence"
    if score_spread <= 1e-8:
        return "flat_scores"
    if top_10_lift >= 2.0 and normalized_gains_auc >= 0.55:
        return "concentrated_ranking"
    if top_20_lift >= 1.30 and top_20_capture >= 0.25 and normalized_gains_auc >= 0.20:
        return "useful_ranking"
    if top_20_lift < 1.05 or normalized_gains_auc < 0.05:
        return "diffuse_ranking"
    return "rank_lift_review"


def _next_step(verdict: str) -> str:
    if verdict == "concentrated_ranking":
        return "Use the gains table to choose a review budget, then validate top-decile lift on fresh rows."
    if verdict == "useful_ranking":
        return "Compare rank lift with capacity planning and threshold tradeoffs before assigning action queues."
    if verdict == "no_positive_evidence":
        return "Load labeled rows with observed positives before relying on rank-lift evidence."
    if verdict == "flat_scores":
        return "Improve feature signal or model training before using ranked review queues."
    if verdict == "diffuse_ranking":
        return "Treat the score ranking as weak; inspect labels, features, and model family before promotion."
    return "Review the gains table and rerun with more validation rows before operational use."


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = str(summary.get("verdict", "rank_lift_review"))
    priority = "medium"
    score = 60.0
    category = "ranking"
    title = "Review rank-lift evidence"
    if verdict == "concentrated_ranking":
        priority = "low"
        score = 40.0
        title = "Validate concentrated top-ranked signal"
    elif verdict in {"diffuse_ranking", "flat_scores"}:
        priority = "high"
        score = 82.0
        title = "Improve ranking before action queues"
    elif verdict == "no_positive_evidence":
        priority = "medium"
        score = 65.0
        category = "evidence"
        title = "Add positive examples before rank-lift review"
    return [
        {
            "rank": 1,
            "priority_score": score,
            "priority": priority,
            "category": category,
            "title": title,
            "reason": f"Rank-lift verdict is {verdict}.",
            "action": summary.get("recommended_next_step"),
        }
    ]
