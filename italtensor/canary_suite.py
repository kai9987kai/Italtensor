from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .schema_guard import check_vector_against_schema


DEFAULT_MAX_CANARIES = 32
DEFAULT_MIN_PROBABILITY_MARGIN = 0.05


def run_canary_suite(
    model: Any,
    examples: Sequence[dict[str, Any]],
    *,
    input_dim: int,
    threshold: float = 0.5,
    preprocessor: Any | None = None,
    schema_guard_report: dict[str, Any] | None = None,
    max_examples: int = DEFAULT_MAX_CANARIES,
    min_probability_margin: float = DEFAULT_MIN_PROBABILITY_MARGIN,
) -> dict[str, Any]:
    """Run deterministic prediction probes against the active model."""
    if model is None:
        raise ValueError("Canary suite needs an active model.")
    if input_dim <= 0:
        raise ValueError("Canary suite needs a positive input dimension.")
    if not np.isfinite(float(threshold)):
        raise ValueError("Canary threshold must be finite.")
    if not np.isfinite(float(min_probability_margin)) or float(min_probability_margin) < 0.0:
        raise ValueError("Canary probability margin must be a finite non-negative number.")
    sanitized = _sanitize_examples(examples, input_dim=input_dim, max_examples=max_examples)

    rows: list[dict[str, Any]] = []
    for example in sanitized:
        raw = np.asarray(example["features"], dtype=np.float32).reshape(1, -1)
        prepared = preprocessor.transform(raw) if preprocessor is not None else raw
        probability = float(predict_probability(model, prepared)[0])
        if not np.isfinite(probability):
            raise ValueError(f"Canary example '{example['name']}' produced a non-finite probability.")
        predicted_label = 1 if probability >= threshold else 0
        expected_label = example["expected_label"]
        margin = abs(probability - threshold)
        schema_status = "not_run"
        schema_warning_count = 0
        schema_failure_count = 0
        schema_failures: list[dict[str, Any]] = []
        schema_warnings: list[dict[str, Any]] = []
        if schema_guard_report is not None:
            guard = check_vector_against_schema(example["features"], schema_guard_report)
            schema_status = str(guard.get("status", "not_run"))
            schema_warning_count = int(guard.get("warning_count", 0) or 0)
            schema_failure_count = int(guard.get("failure_count", 0) or 0)
            schema_failures = list(guard.get("failures", [])[:3])
            schema_warnings = list(guard.get("warnings", [])[:3])

        if expected_label is None:
            passed = None
            status = "informational"
        else:
            passed = predicted_label == expected_label
            status = "pass" if passed else "fail"
        if status == "pass" and margin < min_probability_margin:
            status = "review"
        if schema_failure_count:
            status = "schema_fail" if status != "fail" else status
        elif schema_warning_count and status == "pass":
            status = "schema_warn"

        rows.append(
            {
                "index": int(example["index"]),
                "name": example["name"],
                "features": [float(value) for value in example["features"]],
                "expected_label": expected_label,
                "predicted_label": int(predicted_label),
                "probability": probability,
                "threshold": float(threshold),
                "margin_to_threshold": float(margin),
                "confidence": float(max(probability, 1.0 - probability)),
                "passed": passed,
                "status": status,
                "schema_status": schema_status,
                "schema_warning_count": int(schema_warning_count),
                "schema_failure_count": int(schema_failure_count),
                "schema_failures": schema_failures,
                "schema_warnings": schema_warnings,
            }
        )

    summary = _summarize(
        rows,
        threshold=float(threshold),
        min_probability_margin=float(min_probability_margin),
        truncated_count=max(0, len(examples) - len(sanitized)),
    )
    return {
        "input_dim": int(input_dim),
        "threshold": float(threshold),
        "max_examples": int(max_examples),
        "min_probability_margin": float(min_probability_margin),
        "example_count": int(len(rows)),
        "truncated_count": int(summary["truncated_count"]),
        "summary": summary,
        "examples": rows,
    }


def format_canary_suite_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    pass_rate = summary.get("pass_rate")
    pass_rate_text = "-" if pass_rate is None else f"{float(pass_rate):.3f}"
    min_margin = summary.get("min_probability_margin_observed")
    min_margin_text = "-" if min_margin is None else f"{float(min_margin):.4f}"
    return (
        "Canary suite: "
        f"verdict={summary.get('verdict', '-')}, "
        f"passed={int(summary.get('passed_count', 0))}/{int(summary.get('checked_count', 0))}, "
        f"failed={int(summary.get('failed_count', 0))}, "
        f"review={int(summary.get('review_count', 0))}, "
        f"informational={int(summary.get('informational_count', 0))}, "
        f"pass_rate={pass_rate_text}, "
        f"min_margin={min_margin_text}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def _sanitize_examples(
    examples: Sequence[dict[str, Any]],
    *,
    input_dim: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    if not examples:
        raise ValueError("Canary suite needs at least one prediction example.")
    if max_examples <= 0:
        raise ValueError("Canary max_examples must be positive.")
    sanitized: list[dict[str, Any]] = []
    for index, example in enumerate(list(examples)[:max_examples]):
        if not isinstance(example, dict):
            raise ValueError("Canary examples must be JSON objects.")
        try:
            features = np.asarray(example.get("features"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Canary example features must be numeric.") from exc
        if features.shape[0] != input_dim:
            raise ValueError(f"Canary example {index + 1} must contain {input_dim} feature(s).")
        if not np.all(np.isfinite(features)):
            raise ValueError("Canary example features must be finite numbers.")
        name = str(example.get("name") or f"Canary {index + 1}").strip() or f"Canary {index + 1}"
        sanitized.append(
            {
                "index": int(index),
                "name": name,
                "features": [float(value) for value in features],
                "expected_label": _parse_expected_label(example.get("expected_label")),
            }
        )
    return sanitized


def _parse_expected_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("Canary expected_label must be 0, 1, or null.")
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if text not in {"0", "1"}:
            raise ValueError("Canary expected_label must be 0, 1, or null.")
        parsed = int(text)
    else:
        raise ValueError("Canary expected_label must be 0, 1, or null.")
    if parsed not in {0, 1}:
        raise ValueError("Canary expected_label must be 0, 1, or null.")
    return parsed


def _summarize(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    min_probability_margin: float,
    truncated_count: int,
) -> dict[str, Any]:
    checked = [row for row in rows if row["expected_label"] is not None]
    informational = [row for row in rows if row["expected_label"] is None]
    failed = [row for row in checked if row["passed"] is False]
    passed = [row for row in checked if row["passed"] is True]
    low_margin = [row for row in rows if float(row["margin_to_threshold"]) < min_probability_margin]
    schema_failures = [row for row in rows if int(row.get("schema_failure_count", 0) or 0) > 0]
    schema_warnings = [row for row in rows if int(row.get("schema_warning_count", 0) or 0) > 0]
    review = [
        row
        for row in rows
        if row["status"] in {"review", "schema_warn", "schema_fail"} and row not in failed
    ]
    if failed or schema_failures:
        verdict = "canary_fail"
        next_step = "Investigate failing canary examples before using this model as the active candidate."
    elif not checked:
        verdict = "no_checkable_canaries"
        next_step = "Add expected labels to preset prediction examples to turn review probes into regression checks."
    elif low_margin or schema_warnings:
        verdict = "canary_review"
        next_step = "Review low-margin or schema-warning canaries and decide whether to adjust the threshold or data contract."
    else:
        verdict = "canary_pass"
        next_step = "Save the canary report with the model bundle and rerun it after model changes."
    pass_rate = None if not checked else len(passed) / len(checked)
    margins = [float(row["margin_to_threshold"]) for row in rows]
    return {
        "verdict": verdict,
        "checked_count": int(len(checked)),
        "passed_count": int(len(passed)),
        "failed_count": int(len(failed)),
        "informational_count": int(len(informational)),
        "review_count": int(len(review)),
        "low_margin_count": int(len(low_margin)),
        "schema_warning_count": int(len(schema_warnings)),
        "schema_failure_count": int(len(schema_failures)),
        "pass_rate": None if pass_rate is None else round(float(pass_rate), 6),
        "threshold": float(threshold),
        "required_probability_margin": float(min_probability_margin),
        "min_probability_margin_observed": None if not margins else min(margins),
        "truncated_count": int(truncated_count),
        "recommended_next_step": next_step,
    }
