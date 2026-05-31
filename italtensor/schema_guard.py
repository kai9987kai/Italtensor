from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


EPSILON = 1e-12


def run_schema_guard(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    max_features: int = 16,
) -> dict[str, Any]:
    """Infer a lightweight numeric feature contract from the loaded dataset."""
    x = _validate_features(features)
    names = _feature_names(feature_names, x.shape[1])
    rows = [_feature_contract(x[:, index], index, names[index]) for index in range(x.shape[1])]
    rows.sort(
        key=lambda row: (
            -float(row["risk_score"]),
            -len(row["risk_flags"]),
            int(row["feature_index"]),
        )
    )
    summary = _summary(rows, x.shape[0])
    report: dict[str, Any] = {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dataset_fingerprint": schema_guard_dataset_fingerprint(x),
        "summary": summary,
        "features": rows[: max(1, int(max_features))],
        "contract": {
            "feature_count": int(x.shape[1]),
            "feature_names": names,
            "checks": [
                "finite_numeric",
                "observed_min_max",
                "soft_1_99_percentile_bounds",
                "robust_median_mad_bounds",
                "low_cardinality_seen_values",
            ],
        },
        "recommendations": _recommendations(summary),
    }
    if labels is not None:
        report["class_counts"] = _class_counts(labels, x.shape[0])
    return report


def check_vector_against_schema(
    vector: Sequence[float] | np.ndarray,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Compare one raw prediction vector with a Schema Guard report."""
    input_dim = int(report.get("input_dim", 0) or 0)
    try:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Schema guard prediction vector must be numeric.") from exc
    if input_dim and values.shape[0] != input_dim:
        raise ValueError(f"Schema guard expected {input_dim} feature(s), got {values.shape[0]}.")
    if values.shape[0] == 0:
        raise ValueError("Schema guard prediction vector must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Schema guard prediction vector must be finite.")

    feature_rows = {int(row.get("feature_index", -1)): row for row in report.get("features", [])}
    warnings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        row = feature_rows.get(index)
        if not row:
            continue
        issues = _vector_issues(float(value), row)
        target = failures if any(issue["severity"] == "fail" for issue in issues) else warnings
        for issue in issues:
            target.append(
                {
                    "feature_index": int(index),
                    "feature_name": row.get("feature_name", f"x{index + 1}"),
                    "value": float(value),
                    **issue,
                }
            )
    status = "fail" if failures else ("warn" if warnings else "pass")
    return {
        "status": status,
        "warning_count": int(len(warnings)),
        "failure_count": int(len(failures)),
        "warnings": warnings,
        "failures": failures,
        "summary": (
            f"Schema guard vector check: status={status}, "
            f"warnings={len(warnings)}, failures={len(failures)}"
        ),
    }


def format_schema_guard_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Schema guard: "
        f"risk={summary.get('risk_level', '-')}, "
        f"score={float(summary.get('readiness_score', 0.0)):.1f}/100, "
        f"constant={int(summary.get('constant_feature_count', 0))}, "
        f"low_cardinality={int(summary.get('low_cardinality_feature_count', 0))}, "
        f"outlier_features={int(summary.get('outlier_feature_count', 0))}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def schema_guard_dataset_fingerprint(features: Sequence[Sequence[float]] | np.ndarray) -> str:
    x = _validate_features(features)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(value) for value in x.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _validate_features(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Schema guard features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("Schema guard features must be a 2D array.")
    if x.shape[0] < 2:
        raise ValueError("Schema guard needs at least two rows.")
    if x.shape[1] == 0:
        raise ValueError("Schema guard needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Schema guard features must be finite numbers.")
    return x


def _feature_names(feature_names: Sequence[str] | None, input_dim: int) -> list[str]:
    if feature_names is None:
        return [f"x{index + 1}" for index in range(input_dim)]
    names = [str(name).strip() or f"x{index + 1}" for index, name in enumerate(feature_names)]
    if len(names) != input_dim:
        raise ValueError("Schema guard feature_names length must match input dimension.")
    return names


def _class_counts(labels: Sequence[int] | np.ndarray, sample_count: int) -> dict[str, int] | None:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if y.shape[0] != sample_count or not np.all(np.isfinite(y)):
        return None
    if not np.all((y == 0.0) | (y == 1.0)):
        return None
    return {"0": int(np.sum(y == 0.0)), "1": int(np.sum(y == 1.0))}


def _feature_contract(values: np.ndarray, index: int, name: str) -> dict[str, Any]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    q01, q05, q25, median, q75, q95, q99 = [float(value) for value in quantiles]
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    mean = float(np.mean(values))
    std = float(np.std(values))
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(values - median)))
    robust_scale = 1.4826 * mad if mad > EPSILON else iqr / 1.349 if iqr > EPSILON else std
    robust_lower = float(median - 6.0 * max(robust_scale, EPSILON))
    robust_upper = float(median + 6.0 * max(robust_scale, EPSILON))
    soft_lower = q01
    soft_upper = q99
    unique_values = np.unique(values)
    unique_count = int(unique_values.shape[0])
    unique_ratio = float(unique_count / values.shape[0])
    outside_robust = (values < robust_lower) | (values > robust_upper)
    outside_soft = (values < soft_lower) | (values > soft_upper)
    outlier_count = int(np.sum(outside_robust))
    soft_outlier_count = int(np.sum(outside_soft))
    flags = _risk_flags(
        values=values,
        std=std,
        iqr=iqr,
        mad=mad,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        outlier_count=outlier_count,
        soft_outlier_count=soft_outlier_count,
    )
    return {
        "feature_index": int(index),
        "feature_name": name,
        "min": minimum,
        "q01": q01,
        "q05": q05,
        "q25": q25,
        "median": median,
        "q75": q75,
        "q95": q95,
        "q99": q99,
        "max": maximum,
        "mean": mean,
        "std": std,
        "mad": mad,
        "iqr": iqr,
        "soft_lower": soft_lower,
        "soft_upper": soft_upper,
        "robust_lower": robust_lower,
        "robust_upper": robust_upper,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "observed_values": _observed_values(unique_values),
        "outlier_count": outlier_count,
        "outlier_fraction": float(outlier_count / values.shape[0]),
        "soft_outlier_count": soft_outlier_count,
        "risk_flags": flags,
        "risk_score": _risk_score(flags, outlier_count / values.shape[0]),
    }


def _risk_flags(
    *,
    values: np.ndarray,
    std: float,
    iqr: float,
    mad: float,
    unique_count: int,
    unique_ratio: float,
    outlier_count: int,
    soft_outlier_count: int,
) -> list[str]:
    flags: list[str] = []
    if std <= EPSILON or float(np.min(values)) == float(np.max(values)):
        flags.append("constant_feature")
    elif unique_count <= min(3, max(2, values.shape[0] // 20)) or unique_ratio <= 0.03:
        flags.append("near_constant_feature")
    if unique_count <= min(20, max(4, values.shape[0] // 4)) and unique_ratio <= 0.20:
        flags.append("low_cardinality_numeric")
    if outlier_count:
        flags.append("robust_range_outliers")
    if soft_outlier_count >= max(2, int(np.ceil(0.02 * values.shape[0]))):
        flags.append("heavy_tail")
    if iqr <= EPSILON and mad <= EPSILON and std > EPSILON:
        flags.append("mostly_constant_with_spikes")
    return flags


def _observed_values(unique_values: np.ndarray) -> list[float] | None:
    if unique_values.shape[0] > 20:
        return None
    return [float(value) for value in unique_values.tolist()]


def _risk_score(flags: list[str], outlier_fraction: float) -> float:
    score = 0.0
    if "constant_feature" in flags:
        score += 32.0
    if "near_constant_feature" in flags:
        score += 18.0
    if "low_cardinality_numeric" in flags:
        score += 10.0
    if "robust_range_outliers" in flags:
        score += min(24.0, 8.0 + 80.0 * outlier_fraction)
    if "heavy_tail" in flags:
        score += 8.0
    if "mostly_constant_with_spikes" in flags:
        score += 16.0
    return float(score)


def _summary(rows: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    constant_count = sum("constant_feature" in row["risk_flags"] for row in rows)
    near_constant_count = sum("near_constant_feature" in row["risk_flags"] for row in rows)
    low_cardinality_count = sum("low_cardinality_numeric" in row["risk_flags"] for row in rows)
    outlier_count = sum("robust_range_outliers" in row["risk_flags"] for row in rows)
    spike_count = sum("mostly_constant_with_spikes" in row["risk_flags"] for row in rows)
    max_outlier_fraction = max((float(row["outlier_fraction"]) for row in rows), default=0.0)
    penalties = (
        14.0 * constant_count
        + 8.0 * near_constant_count
        + 4.0 * low_cardinality_count
        + 8.0 * outlier_count
        + 10.0 * spike_count
        + (8.0 if sample_count < 20 else 0.0)
    )
    readiness = max(0.0, 100.0 - penalties)
    if readiness < 65.0 or constant_count >= 2 or spike_count >= 2:
        risk = "high"
    elif readiness < 84.0 or constant_count or outlier_count:
        risk = "medium"
    else:
        risk = "low"
    actions = _actions(
        constant_count=constant_count,
        near_constant_count=near_constant_count,
        low_cardinality_count=low_cardinality_count,
        outlier_count=outlier_count,
        spike_count=spike_count,
        sample_count=sample_count,
    )
    return {
        "risk_level": risk,
        "readiness_score": round(float(readiness), 1),
        "constant_feature_count": int(constant_count),
        "near_constant_feature_count": int(near_constant_count),
        "low_cardinality_feature_count": int(low_cardinality_count),
        "outlier_feature_count": int(outlier_count),
        "spike_feature_count": int(spike_count),
        "max_outlier_fraction": float(max_outlier_fraction),
        "top_risk_feature": None if not rows else int(rows[0]["feature_index"]),
        "recommended_next_step": actions[0] if actions else "Save this schema with the model and compare future batches against it.",
        "top_actions": actions,
    }


def _actions(
    *,
    constant_count: int,
    near_constant_count: int,
    low_cardinality_count: int,
    outlier_count: int,
    spike_count: int,
    sample_count: int,
) -> list[str]:
    actions: list[str] = []
    if constant_count:
        actions.append("Remove or explain constant feature columns before final training.")
    if spike_count:
        actions.append("Inspect mostly constant features with spikes for data-entry or sensor faults.")
    if outlier_count:
        actions.append("Review robust-range outliers and decide whether to cap, fix, or preserve them.")
    if low_cardinality_count:
        actions.append("Document low-cardinality numeric codes so future unseen values can be flagged.")
    if near_constant_count:
        actions.append("Check near-constant columns for dead sensors or unused one-hot codes.")
    if sample_count < 20:
        actions.append("Collect more rows before treating quantile bounds as stable.")
    return actions[:6]


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    actions = summary.get("top_actions") or []
    recs = []
    risk = str(summary.get("risk_level", "low"))
    for rank, action in enumerate(actions, start=1):
        recs.append(
            {
                "rank": rank,
                "priority": "high" if risk == "high" and rank == 1 else "medium",
                "category": "schema",
                "title": "Feature contract action",
                "action": action,
            }
        )
    return recs


def _vector_issues(value: float, row: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if value < float(row["min"]) or value > float(row["max"]):
        issues.append(
            {
                "severity": "fail",
                "kind": "outside_observed_range",
                "message": f"value is outside observed range [{row['min']:.6g}, {row['max']:.6g}]",
            }
        )
    elif value < float(row["soft_lower"]) or value > float(row["soft_upper"]):
        issues.append(
            {
                "severity": "warn",
                "kind": "outside_soft_quantile_range",
                "message": f"value is outside 1-99% range [{row['soft_lower']:.6g}, {row['soft_upper']:.6g}]",
            }
        )
    observed = row.get("observed_values")
    if observed is not None and "low_cardinality_numeric" in row.get("risk_flags", []):
        if not any(abs(value - float(item)) <= 1e-9 for item in observed):
            issues.append(
                {
                    "severity": "fail",
                    "kind": "unseen_low_cardinality_value",
                    "message": "value was not seen for this low-cardinality numeric feature",
                }
            )
    return issues
