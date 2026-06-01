from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .experiments import probability_diagnostics
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_calibration_slice_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    max_features: int = 12,
    bins: int = 4,
    min_count: int | None = None,
    n_probability_bins: int = 6,
) -> dict[str, Any]:
    """Rank raw-feature ranges where model probabilities are locally miscalibrated."""
    if model is None:
        raise ValueError("Calibration slices need an active model.")
    x, y = _validate_inputs(features, labels)
    max_features = max(1, int(max_features))
    bins = max(2, int(bins))
    n_probability_bins = max(2, int(n_probability_bins))
    minimum = int(min_count if min_count is not None else max(8, round(x.shape[0] * 0.08)))
    minimum = max(2, minimum)

    probabilities = _predict_probabilities(model, x, preprocessor)
    base = _calibration_metrics(y, probabilities, n_probability_bins)
    slice_rows: list[dict[str, Any]] = []
    for feature_index in range(min(x.shape[1], max_features)):
        values = x[:, feature_index]
        if np.allclose(values, values[0]):
            continue
        for left, right, is_last in _quantile_ranges(values, bins):
            mask = (values >= left) & (values <= right) if is_last else (values >= left) & (values < right)
            count = int(np.sum(mask))
            if count < minimum:
                continue
            slice_rows.append(
                _slice_row(
                    feature_index,
                    left,
                    right,
                    count,
                    total_count=x.shape[0],
                    labels=y[mask],
                    probabilities=probabilities[mask],
                    base=base,
                    n_probability_bins=n_probability_bins,
                )
            )

    slice_rows.sort(
        key=lambda item: (
            -float(item["weighted_calibration_impact"]),
            -float(item["absolute_confidence_gap"]),
            -float(item["expected_calibration_error"]),
            -int(item["count"]),
            int(item["feature_index"]),
            float(item["left"]),
        )
    )
    top_slices = slice_rows[:12]
    summary = _summary(top_slices, slice_rows)
    recommendations = _recommendations(summary)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dataset_fingerprint": calibration_slice_dataset_fingerprint(x, y),
        "max_features": max_features,
        "bin_count": bins,
        "min_count": minimum,
        "n_probability_bins": n_probability_bins,
        "base": base,
        "slices": top_slices,
        "summary": summary,
        "recommendations": recommendations,
    }


def format_calibration_slice_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Calibration slices: "
        f"risk={summary.get('risk_level', '-')}, "
        f"slices={int(summary.get('slice_count', 0))}, "
        f"worst={summary.get('worst_slice', 'none')}, "
        f"gap={float(summary.get('max_absolute_confidence_gap', 0.0)):.4f}, "
        f"impact={float(summary.get('max_weighted_calibration_impact', 0.0)):.4f}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def calibration_slice_dataset_fingerprint(
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
        raise ValueError("Calibration slices features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Calibration slices features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Calibration slices feature and label counts do not match.")
    if x.shape[0] < 1:
        raise ValueError("Calibration slices need at least one sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Calibration slices features must be finite numbers.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration slices labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Calibration slices labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Calibration slices labels must be binary 0/1.")
    return y.astype(np.int32)


def _predict_probabilities(
    model: Any,
    x: np.ndarray,
    preprocessor: FeatureStandardizer | None,
) -> np.ndarray:
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Calibration slices preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    return np.clip(probabilities, 0.0, 1.0)


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, n_probability_bins: int) -> dict[str, float]:
    diagnostics = probability_diagnostics(labels, probabilities, n_bins=n_probability_bins)
    mean_probability = float(diagnostics.get("mean_probability", 0.0))
    label_prevalence = float(diagnostics.get("label_prevalence", 0.0))
    signed_gap = float(mean_probability - label_prevalence)
    return {
        "brier_score": float(diagnostics.get("brier_score", 0.0)),
        "log_loss": float(diagnostics.get("log_loss", 0.0)),
        "expected_calibration_error": float(diagnostics.get("expected_calibration_error", 0.0)),
        "max_calibration_error": float(diagnostics.get("max_calibration_error", 0.0)),
        "mean_probability": mean_probability,
        "label_prevalence": label_prevalence,
        "signed_confidence_gap": signed_gap,
        "absolute_confidence_gap": abs(signed_gap),
        "calibration_direction": _direction(signed_gap),
    }


def _slice_row(
    feature_index: int,
    left: float,
    right: float,
    count: int,
    *,
    total_count: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
    base: dict[str, float],
    n_probability_bins: int,
) -> dict[str, Any]:
    metrics = _calibration_metrics(labels, probabilities, n_probability_bins)
    coverage = float(count / max(1, total_count))
    weighted_impact = coverage * float(metrics["absolute_confidence_gap"])
    return {
        "feature_index": int(feature_index),
        "left": float(left),
        "right": float(right),
        "count": int(count),
        "coverage": coverage,
        "mean_probability": float(metrics["mean_probability"]),
        "label_prevalence": float(metrics["label_prevalence"]),
        "signed_confidence_gap": float(metrics["signed_confidence_gap"]),
        "absolute_confidence_gap": float(metrics["absolute_confidence_gap"]),
        "calibration_direction": metrics["calibration_direction"],
        "expected_calibration_error": float(metrics["expected_calibration_error"]),
        "max_calibration_error": float(metrics["max_calibration_error"]),
        "brier_score": float(metrics["brier_score"]),
        "brier_delta": float(metrics["brier_score"] - float(base.get("brier_score", 0.0))),
        "log_loss": float(metrics["log_loss"]),
        "weighted_calibration_impact": float(weighted_impact),
        "risk_flags": _risk_flags(metrics, weighted_impact),
    }


def _summary(top_slices: list[dict[str, Any]], all_slices: list[dict[str, Any]]) -> dict[str, Any]:
    worst = top_slices[0] if top_slices else None
    max_gap = max((float(item["absolute_confidence_gap"]) for item in all_slices), default=0.0)
    max_ece = max((float(item["expected_calibration_error"]) for item in all_slices), default=0.0)
    max_impact = max((float(item["weighted_calibration_impact"]) for item in all_slices), default=0.0)
    high_risk_count = sum(1 for item in all_slices if "high_local_miscalibration" in item.get("risk_flags", []))
    direction_counts = {
        "overconfident": sum(1 for item in all_slices if item.get("calibration_direction") == "overconfident"),
        "underconfident": sum(1 for item in all_slices if item.get("calibration_direction") == "underconfident"),
        "aligned": sum(1 for item in all_slices if item.get("calibration_direction") == "aligned"),
    }
    risk_level = _risk_level(max_gap, max_ece, max_impact, high_risk_count)
    return {
        "risk_level": risk_level,
        "slice_count": int(len(all_slices)),
        "high_risk_slice_count": int(high_risk_count),
        "worst_slice": _slice_name(worst) if worst else "none",
        "worst_feature": int(worst["feature_index"]) if worst else None,
        "worst_direction": worst.get("calibration_direction") if worst else None,
        "max_absolute_confidence_gap": float(max_gap),
        "max_expected_calibration_error": float(max_ece),
        "max_weighted_calibration_impact": float(max_impact),
        "overconfident_slice_count": int(direction_counts["overconfident"]),
        "underconfident_slice_count": int(direction_counts["underconfident"]),
        "aligned_slice_count": int(direction_counts["aligned"]),
        "recommendation": _next_step(risk_level, worst),
    }


def _quantile_ranges(values: np.ndarray, bins: int) -> list[tuple[float, float, bool]]:
    quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 2:
        return []
    ranges: list[tuple[float, float, bool]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if float(left) == float(right):
            continue
        ranges.append((float(left), float(right), index == edges.size - 2))
    return ranges


def _direction(signed_gap: float) -> str:
    if signed_gap >= 0.03:
        return "overconfident"
    if signed_gap <= -0.03:
        return "underconfident"
    return "aligned"


def _risk_flags(metrics: dict[str, float], weighted_impact: float) -> list[str]:
    flags: list[str] = []
    if float(metrics["absolute_confidence_gap"]) >= 0.20 or float(metrics["expected_calibration_error"]) >= 0.18:
        flags.append("high_local_miscalibration")
    if weighted_impact >= 0.05:
        flags.append("high_weighted_impact")
    if abs(float(metrics["brier_score"])) >= 0.30:
        flags.append("high_brier")
    return flags


def _risk_level(max_gap: float, max_ece: float, max_impact: float, high_risk_count: int) -> str:
    if max_gap >= 0.25 or max_ece >= 0.22 or max_impact >= 0.08 or high_risk_count >= 3:
        return "high"
    if max_gap >= 0.15 or max_ece >= 0.14 or max_impact >= 0.04 or high_risk_count:
        return "medium"
    return "low"


def _next_step(risk_level: str, worst: dict[str, Any] | None) -> str:
    if not worst:
        return "No calibration slice action is available; collect more rows or lower the slice minimum for exploration."
    name = _slice_name(worst)
    if risk_level == "high":
        return f"Review {name}, then run Calibration repair or collect more representative rows for that slice."
    if risk_level == "medium":
        return f"Inspect {name} before relying on probabilities for local decisions."
    return "Keep localized calibration evidence with the model report."


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    risk = str(summary.get("risk_level", "low"))
    recs: list[dict[str, Any]] = []

    def add(score: float, priority: str, category: str, title: str, action: str) -> None:
        recs.append(
            {
                "priority_score": float(score),
                "priority": priority,
                "category": category,
                "title": title,
                "action": action,
            }
        )

    if risk == "high":
        add(
            92.0,
            "high",
            "calibration",
            "Repair or revalidate localized miscalibration",
            str(summary.get("recommendation") or "Run Calibration repair and inspect the worst slice."),
        )
    elif risk == "medium":
        add(
            68.0,
            "medium",
            "slice_review",
            "Review local probability trust",
            str(summary.get("recommendation") or "Inspect the worst calibration slice before deployment."),
        )
    else:
        add(
            30.0,
            "low",
            "evidence",
            "Retain localized calibration evidence",
            "Export the report or model sidecar so local calibration evidence is retained.",
        )
    for index, item in enumerate(sorted(recs, key=lambda row: -float(row["priority_score"])), start=1):
        item["rank"] = index
    return recs


def _slice_name(item: dict[str, Any] | None) -> str:
    if not item:
        return "none"
    return f"x{int(item['feature_index']) + 1}[{float(item['left']):.4g}, {float(item['right']):.4g}]"
