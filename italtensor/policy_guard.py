from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability


DEFAULT_MAX_PAIRS_PER_CHECK = 64
DEFAULT_TOLERANCE = 1e-4
DEFAULT_STEP_FRACTION = 0.20


def run_policy_guard(
    model: Any,
    reference_features: Sequence[Sequence[float]] | np.ndarray,
    policy_checks: Sequence[dict[str, Any]],
    *,
    input_dim: int,
    preprocessor: Any | None = None,
    threshold: float = 0.5,
    max_pairs_per_check: int = DEFAULT_MAX_PAIRS_PER_CHECK,
    tolerance: float = DEFAULT_TOLERANCE,
    step_fraction: float = DEFAULT_STEP_FRACTION,
) -> dict[str, Any]:
    """Probe whether an active model respects preset-supplied monotonic policy checks."""
    if model is None:
        raise ValueError("Policy guard needs an active model.")
    if input_dim <= 0:
        raise ValueError("Policy guard needs a positive input dimension.")
    if max_pairs_per_check <= 0:
        raise ValueError("Policy guard max_pairs_per_check must be positive.")
    if not np.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("Policy guard tolerance must be a finite non-negative number.")
    if not np.isfinite(float(step_fraction)) or float(step_fraction) <= 0.0:
        raise ValueError("Policy guard step_fraction must be a finite positive number.")
    x = _validate_features(reference_features, input_dim)
    checks = _sanitize_checks(policy_checks, input_dim)

    results = [
        _run_one_check(
            model,
            x,
            check,
            preprocessor=preprocessor,
            threshold=float(threshold),
            max_pairs_per_check=max_pairs_per_check,
            tolerance=float(tolerance),
            step_fraction=float(step_fraction),
        )
        for check in checks
    ]
    summary = _summary(results, tolerance=float(tolerance))
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(input_dim),
        "threshold": float(threshold),
        "max_pairs_per_check": int(max_pairs_per_check),
        "tolerance": float(tolerance),
        "step_fraction": float(step_fraction),
        "summary": summary,
        "checks": results,
        "recommendations": _recommendations(summary),
    }


def format_policy_guard_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Policy guard: "
        f"verdict={summary.get('verdict', '-')}, "
        f"checks={int(summary.get('check_count', 0))}, "
        f"violations={int(summary.get('violation_count', 0))}/{int(summary.get('pair_count', 0))}, "
        f"rate={float(summary.get('violation_rate', 0.0)):.3f}, "
        f"max={float(summary.get('max_violation', 0.0)):.4f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def sanitize_policy_checks(
    checks: Sequence[dict[str, Any]] | None,
    input_dim: int,
) -> list[dict[str, Any]]:
    if not checks:
        return []
    return _sanitize_checks(checks, input_dim)


def _validate_features(features: Sequence[Sequence[float]] | np.ndarray, input_dim: int) -> np.ndarray:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Policy guard reference features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Policy guard reference features must be a 2D array.")
    if x.shape[0] < 2:
        raise ValueError("Policy guard needs at least two reference rows.")
    if x.shape[1] != input_dim:
        raise ValueError(f"Policy guard expected {input_dim} feature(s), got {x.shape[1]}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Policy guard reference features must be finite numbers.")
    return x.astype(np.float32)


def _sanitize_checks(checks: Sequence[dict[str, Any]], input_dim: int) -> list[dict[str, Any]]:
    if not checks:
        raise ValueError("Policy guard needs at least one policy check.")
    sanitized: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError("Policy checks must be JSON objects.")
        if check.get("kind", "monotonic") != "monotonic":
            raise ValueError("Policy guard v1 only supports monotonic checks.")
        try:
            feature_index = int(check.get("feature_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Policy check feature_index must be an integer.") from exc
        if feature_index < 0 or feature_index >= input_dim:
            raise ValueError(f"Policy check feature_index {feature_index} is out of bounds.")
        direction = _parse_direction(check.get("direction"))
        name = str(check.get("name") or check.get("feature_name") or f"Policy check {index + 1}").strip()
        feature_name = str(check.get("feature_name") or f"x{feature_index + 1}").strip()
        sanitized.append(
            {
                "kind": "monotonic",
                "name": name or f"Policy check {index + 1}",
                "feature_index": feature_index,
                "feature_name": feature_name or f"x{feature_index + 1}",
                "direction": direction,
                "description": str(check.get("description") or "").strip(),
            }
        )
    return sanitized


def _parse_direction(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return "increasing"
        if int(value) == -1:
            return "decreasing"
    text = str(value).strip().lower()
    aliases = {
        "increase": "increasing",
        "increasing": "increasing",
        "positive": "increasing",
        "+": "increasing",
        "+1": "increasing",
        "decrease": "decreasing",
        "decreasing": "decreasing",
        "negative": "decreasing",
        "-": "decreasing",
        "-1": "decreasing",
    }
    if text not in aliases:
        raise ValueError("Policy check direction must be increasing or decreasing.")
    return aliases[text]


def _run_one_check(
    model: Any,
    x: np.ndarray,
    check: dict[str, Any],
    *,
    preprocessor: Any | None,
    threshold: float,
    max_pairs_per_check: int,
    tolerance: float,
    step_fraction: float,
) -> dict[str, Any]:
    feature_index = int(check["feature_index"])
    column = x[:, feature_index]
    low = float(np.quantile(column, 0.05))
    high = float(np.quantile(column, 0.95))
    span = high - low
    if span <= 1e-8:
        return {
            **check,
            "status": "not_testable",
            "pair_count": 0,
            "violation_count": 0,
            "violation_rate": 0.0,
            "max_violation": 0.0,
            "mean_delta": 0.0,
            "feature_range": [low, high],
            "threshold_crossing_count": 0,
            "violations": [],
            "warning": "Feature has too little observed variation for a monotonic probe.",
        }
    delta = max(span * step_fraction, 1e-8)
    row_indices = _row_indices(x.shape[0], max_pairs_per_check)
    lower_rows: list[np.ndarray] = []
    upper_rows: list[np.ndarray] = []
    source_rows: list[int] = []
    for row_index in row_indices:
        lower_value = float(np.clip(x[row_index, feature_index], low, high - delta))
        upper_value = lower_value + delta
        if upper_value > high + 1e-8:
            continue
        lower = np.asarray(x[row_index], dtype=np.float32).copy()
        upper = np.asarray(x[row_index], dtype=np.float32).copy()
        lower[feature_index] = lower_value
        upper[feature_index] = upper_value
        lower_rows.append(lower)
        upper_rows.append(upper)
        source_rows.append(int(row_index))
    if not lower_rows:
        return {
            **check,
            "status": "not_testable",
            "pair_count": 0,
            "violation_count": 0,
            "violation_rate": 0.0,
            "max_violation": 0.0,
            "mean_delta": 0.0,
            "feature_range": [low, high],
            "threshold_crossing_count": 0,
            "violations": [],
            "warning": "No reference rows could be converted into monotonic probe pairs.",
        }

    lower_array = np.asarray(lower_rows, dtype=np.float32)
    upper_array = np.asarray(upper_rows, dtype=np.float32)
    lower_prob = _predict(model, lower_array, preprocessor)
    upper_prob = _predict(model, upper_array, preprocessor)
    deltas = upper_prob - lower_prob
    if check["direction"] == "increasing":
        violation_sizes = np.maximum(0.0, lower_prob - upper_prob - tolerance)
    else:
        violation_sizes = np.maximum(0.0, upper_prob - lower_prob - tolerance)
    violation_mask = violation_sizes > 0.0
    crossing_count = int(np.sum((lower_prob < threshold) != (upper_prob < threshold)))
    violations = [
        {
            "source_row": int(source_rows[i]),
            "lower_value": float(lower_array[i, feature_index]),
            "upper_value": float(upper_array[i, feature_index]),
            "lower_probability": float(lower_prob[i]),
            "upper_probability": float(upper_prob[i]),
            "probability_delta": float(deltas[i]),
            "violation": float(violation_sizes[i]),
        }
        for i in np.argsort(-violation_sizes)[:5]
        if violation_mask[i]
    ]
    violation_count = int(np.sum(violation_mask))
    pair_count = int(lower_array.shape[0])
    violation_rate = float(violation_count / max(pair_count, 1))
    max_violation = float(np.max(violation_sizes)) if violation_sizes.size else 0.0
    if violation_rate > 0.10 or max_violation >= 0.05:
        status = "fail"
    elif violation_count:
        status = "review"
    else:
        status = "pass"
    return {
        **check,
        "status": status,
        "pair_count": pair_count,
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "max_violation": max_violation,
        "mean_delta": float(np.mean(deltas)) if deltas.size else 0.0,
        "feature_range": [low, high],
        "step": float(delta),
        "threshold_crossing_count": crossing_count,
        "violations": violations,
        "warning": None,
    }


def _predict(model: Any, rows: np.ndarray, preprocessor: Any | None) -> np.ndarray:
    prepared = preprocessor.transform(rows) if preprocessor is not None else rows
    probabilities = predict_probability(model, prepared).reshape(-1)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Policy guard model produced non-finite probabilities.")
    return probabilities.astype(np.float64)


def _row_indices(row_count: int, max_pairs: int) -> np.ndarray:
    if row_count <= max_pairs:
        return np.arange(row_count, dtype=np.int32)
    return np.linspace(0, row_count - 1, num=max_pairs, dtype=np.int32)


def _summary(results: list[dict[str, Any]], *, tolerance: float) -> dict[str, Any]:
    pair_count = sum(int(item.get("pair_count", 0) or 0) for item in results)
    violation_count = sum(int(item.get("violation_count", 0) or 0) for item in results)
    not_testable_count = sum(item.get("status") == "not_testable" for item in results)
    failed_count = sum(item.get("status") == "fail" for item in results)
    review_count = sum(item.get("status") == "review" for item in results)
    max_violation = max((float(item.get("max_violation", 0.0) or 0.0) for item in results), default=0.0)
    violation_rate = float(violation_count / max(pair_count, 1))
    worst = max(results, key=lambda item: float(item.get("max_violation", 0.0) or 0.0), default=None)
    if failed_count:
        verdict = "policy_fail"
        next_step = "Investigate monotonic policy violations before promotion or constrain/retrain the model."
    elif review_count or not_testable_count:
        verdict = "policy_review"
        next_step = "Review weak or untestable policy checks and decide whether to add data or adjust constraints."
    else:
        verdict = "policy_pass"
        next_step = "Save the policy-guard report with the model and rerun it after model changes."
    return {
        "verdict": verdict,
        "check_count": int(len(results)),
        "pair_count": int(pair_count),
        "violation_count": int(violation_count),
        "violation_rate": violation_rate,
        "failed_check_count": int(failed_count),
        "review_check_count": int(review_count),
        "not_testable_check_count": int(not_testable_count),
        "max_violation": max_violation,
        "worst_check": None if worst is None else worst.get("name"),
        "tolerance": float(tolerance),
        "recommended_next_step": next_step,
    }


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = summary.get("verdict")
    if verdict == "policy_fail":
        return [
            {
                "rank": 1,
                "priority": "high",
                "category": "policy",
                "title": "Resolve monotonic policy violations",
                "action": summary.get("recommended_next_step"),
            }
        ]
    if verdict == "policy_review":
        return [
            {
                "rank": 1,
                "priority": "medium",
                "category": "policy",
                "title": "Review policy guard warnings",
                "action": summary.get("recommended_next_step"),
            }
        ]
    return [
        {
            "rank": 1,
            "priority": "low",
            "category": "policy",
            "title": "Keep policy probes with model evidence",
            "action": "Rerun Policy guard after retraining or loading a new active model.",
        }
    ]
