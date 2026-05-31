from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_shadow_replay(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    window_count: int = 5,
    min_window_size: int = 8,
    max_error_runs: int = 8,
) -> dict[str, Any]:
    """Replay active-model performance across loaded row order."""
    x, y = _validate_inputs(features, labels)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    window_count = max(2, int(window_count))
    min_window_size = max(2, int(min_window_size))
    max_error_runs = max(1, int(max_error_runs))

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Shadow replay preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    predicted = (probabilities >= threshold).astype(np.int32)
    losses = _binary_log_loss(y, probabilities)

    windows = [
        _window_metrics(index, start, end, y, probabilities, predicted, losses)
        for index, start, end in _window_slices(x.shape[0], window_count, min_window_size)
    ]
    baseline = windows[0]
    for window in windows:
        _add_window_deltas(window, baseline)

    degradation_windows = _degradation_windows(windows)
    error_runs = _error_runs(y, predicted, probabilities, losses, max_items=max_error_runs)
    summary = _summary(windows, degradation_windows)
    recommendations = _recommendations(summary, degradation_windows, error_runs)
    summary["recommendation"] = recommendations[0]["action"] if recommendations else None

    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": threshold,
        "window_count": len(windows),
        "min_window_size": min_window_size,
        "row_order_assumption": "loaded_row_order",
        "dataset_fingerprint": shadow_replay_dataset_fingerprint(x, y),
        "summary": summary,
        "windows": windows,
        "segments": windows,
        "checkpoints": _checkpoints(windows),
        "degradation_windows": degradation_windows[:6],
        "worst_windows": degradation_windows[:6],
        "error_runs": error_runs,
        "recommendations": recommendations,
    }


def format_shadow_replay_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Shadow replay: "
        f"verdict={summary.get('verdict', '-')}, "
        f"windows={int(report.get('window_count', 0))}, "
        f"first_f1={float(summary.get('first_window_f1', 0.0)):.4f}, "
        f"last_f1={float(summary.get('last_window_f1', 0.0)):.4f}, "
        f"max_drop={float(summary.get('max_f1_drop', 0.0)):.4f}, "
        f"worst_window={summary.get('worst_window_index', '-')}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def shadow_replay_dataset_fingerprint(
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
        raise ValueError("Shadow replay features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Shadow replay features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Shadow replay feature and label counts do not match.")
    if x.shape[0] < 2:
        raise ValueError("Shadow replay needs at least two samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Shadow replay features must be finite numbers.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Shadow replay labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Shadow replay labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Shadow replay labels must be binary 0/1.")
    return y.astype(np.int32)


def _window_slices(total: int, requested_windows: int, min_window_size: int) -> list[tuple[int, int, int]]:
    if total < 2 * min_window_size:
        count = 2
    else:
        count = min(requested_windows, max(2, total // min_window_size))
    edges = np.linspace(0, total, count + 1)
    edges = np.rint(edges).astype(int)
    edges[0] = 0
    edges[-1] = total
    slices: list[tuple[int, int, int]] = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        if int(end) > int(start):
            slices.append((index, int(start), int(end)))
    return slices


def _window_metrics(
    index: int,
    start: int,
    end: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    losses: np.ndarray,
) -> dict[str, Any]:
    y = labels[start:end]
    p = probabilities[start:end]
    pred = predicted[start:end]
    loss = losses[start:end]
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    count = int(y.shape[0])
    return {
        "window_index": int(index),
        "start_row": int(start),
        "end_row_exclusive": int(end),
        "count": count,
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "prevalence": float(np.mean(y)) if count else 0.0,
        "mean_probability": float(np.mean(p)) if count else 0.0,
        "predicted_positive_rate": float(np.mean(pred == 1)) if count else 0.0,
        "accuracy": float((tp + tn) / count) if count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": float(np.mean((p - y) ** 2)) if count else 0.0,
        "log_loss": float(np.mean(loss)) if count else 0.0,
        "error_rate": float(np.mean(pred != y)) if count else 0.0,
        "false_positive": fp,
        "false_negative": fn,
    }


def _add_window_deltas(window: dict[str, Any], baseline: dict[str, Any]) -> None:
    window["f1_delta_vs_first"] = float(window["f1"] - baseline["f1"])
    window["accuracy_delta_vs_first"] = float(window["accuracy"] - baseline["accuracy"])
    window["brier_delta_vs_first"] = float(window["brier_score"] - baseline["brier_score"])
    window["prevalence_shift_vs_first"] = float(window["prevalence"] - baseline["prevalence"])
    window["probability_shift_vs_first"] = float(window["mean_probability"] - baseline["mean_probability"])
    window["degradation_score"] = float(
        100.0
        * (
            0.45 * max(0.0, -window["f1_delta_vs_first"])
            + 0.25 * max(0.0, -window["accuracy_delta_vs_first"])
            + 0.20 * max(0.0, window["brier_delta_vs_first"])
            + 0.10 * abs(window["prevalence_shift_vs_first"])
        )
    )


def _degradation_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        window
        for window in windows[1:]
        if (
            -float(window["f1_delta_vs_first"]) >= 0.15
            or -float(window["accuracy_delta_vs_first"]) >= 0.15
            or float(window["brier_delta_vs_first"]) >= 0.08
        )
    ]
    selected.sort(
        key=lambda item: (
            -float(item["degradation_score"]),
            float(item["f1"]),
            int(item["window_index"]),
        )
    )
    return selected


def _checkpoints(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "window_index": int(window["window_index"]),
            "end_row_exclusive": int(window["end_row_exclusive"]),
            "cumulative_fraction": float(window["end_row_exclusive"] / windows[-1]["end_row_exclusive"]),
            "f1": float(window["f1"]),
            "accuracy": float(window["accuracy"]),
            "brier_score": float(window["brier_score"]),
            "f1_delta_vs_first": float(window["f1_delta_vs_first"]),
        }
        for window in windows
    ]


def _summary(windows: list[dict[str, Any]], degradation_windows: list[dict[str, Any]]) -> dict[str, Any]:
    first = windows[0]
    last = windows[-1]
    worst = min(windows, key=lambda item: (float(item["f1"]), -float(item["brier_score"]), int(item["window_index"])))
    max_f1_drop = max(0.0, max(-float(window["f1_delta_vs_first"]) for window in windows))
    max_accuracy_drop = max(0.0, max(-float(window["accuracy_delta_vs_first"]) for window in windows))
    max_brier_increase = max(0.0, max(float(window["brier_delta_vs_first"]) for window in windows))
    max_prevalence_shift = max(abs(float(window["prevalence_shift_vs_first"])) for window in windows)
    max_probability_shift = max(abs(float(window["probability_shift_vs_first"])) for window in windows)
    verdict = _verdict(
        windows=windows,
        max_f1_drop=max_f1_drop,
        max_accuracy_drop=max_accuracy_drop,
        max_brier_increase=max_brier_increase,
        degradation_count=len(degradation_windows),
    )
    return {
        "verdict": verdict,
        "first_window_f1": float(first["f1"]),
        "last_window_f1": float(last["f1"]),
        "worst_window_index": int(worst["window_index"]),
        "worst_window_start": int(worst["start_row"]),
        "worst_window_end_exclusive": int(worst["end_row_exclusive"]),
        "worst_window_f1": float(worst["f1"]),
        "worst_window_accuracy": float(worst["accuracy"]),
        "worst_window_brier": float(worst["brier_score"]),
        "last_window_error_rate": float(last["error_rate"]),
        "max_f1_drop": float(max_f1_drop),
        "max_accuracy_drop": float(max_accuracy_drop),
        "max_brier_increase": float(max_brier_increase),
        "max_prevalence_shift": float(max_prevalence_shift),
        "max_probability_shift": float(max_probability_shift),
        "degradation_window_count": int(len(degradation_windows)),
    }


def _verdict(
    *,
    windows: list[dict[str, Any]],
    max_f1_drop: float,
    max_accuracy_drop: float,
    max_brier_increase: float,
    degradation_count: int,
) -> str:
    if any(int(window["count"]) < 5 for window in windows):
        return "thin_ordered_evidence"
    if max_f1_drop >= 0.30 or max_accuracy_drop >= 0.25 or max_brier_increase >= 0.12:
        return "severe_ordered_degradation"
    if max_f1_drop >= 0.15 or max_accuracy_drop >= 0.15 or max_brier_increase >= 0.08 or degradation_count:
        return "ordered_degradation_review"
    return "stable_ordered_replay"


def _error_runs(
    labels: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    losses: np.ndarray,
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    error_mask = predicted != labels
    runs: list[dict[str, Any]] = []
    start: int | None = None
    for index, is_error in enumerate(error_mask.tolist() + [False]):
        if is_error and start is None:
            start = index
        elif not is_error and start is not None:
            end = index
            run_labels = labels[start:end]
            run_predicted = predicted[start:end]
            fp = int(np.sum((run_labels == 0) & (run_predicted == 1)))
            fn = int(np.sum((run_labels == 1) & (run_predicted == 0)))
            confidence = np.where(run_predicted == 1, probabilities[start:end], 1.0 - probabilities[start:end])
            runs.append(
                {
                    "start_row": int(start),
                    "end_row_exclusive": int(end),
                    "length": int(end - start),
                    "false_positive": fp,
                    "false_negative": fn,
                    "mean_loss": float(np.mean(losses[start:end])),
                    "mean_confidence": float(np.mean(confidence)),
                }
            )
            start = None
    runs.sort(key=lambda item: (-int(item["length"]), -float(item["mean_loss"]), int(item["start_row"])))
    return runs[:max_items]


def _recommendations(
    summary: dict[str, Any],
    degradation_windows: list[dict[str, Any]],
    error_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def add(score: float, priority: str, category: str, title: str, reason: str, action: str) -> None:
        recs.append(
            {
                "priority_score": float(score),
                "priority": priority,
                "category": category,
                "title": title,
                "reason": reason,
                "action": action,
            }
        )

    verdict = str(summary.get("verdict", "stable_ordered_replay"))
    if verdict == "severe_ordered_degradation":
        add(
            92.0,
            "high",
            "temporal_validation",
            "Do not trust row-order stability yet",
            f"Max F1 drop={summary['max_f1_drop']:.3f}, max Brier increase={summary['max_brier_increase']:.3f}.",
            "Validate on fresh later rows or retrain with an explicitly time-aware split before promotion.",
        )
    elif verdict == "ordered_degradation_review":
        add(
            74.0,
            "medium",
            "temporal_validation",
            "Review degraded replay windows",
            f"{summary['degradation_window_count']} ordered window(s) degrade against the first window.",
            "Inspect the worst window and compare it with chronological holdout, drift, and error-atlas diagnostics.",
        )
    elif verdict == "thin_ordered_evidence":
        add(
            55.0,
            "low",
            "evidence",
            "Use more ordered rows",
            "At least one replay window has fewer than five rows.",
            "Treat this as a smoke test until more row-ordered evidence is available.",
        )

    if degradation_windows:
        worst = degradation_windows[0]
        add(
            66.0,
            "medium",
            "window_review",
            "Inspect the worst ordered window",
            f"Window {worst['window_index']} rows {worst['start_row']}:{worst['end_row_exclusive']} has F1={worst['f1']:.3f}.",
            "Review row-order context, label mix, and feature shifts in that window before relying on aggregate metrics.",
        )
    if error_runs:
        run = error_runs[0]
        add(
            52.0,
            "low",
            "error_run",
            "Check consecutive error runs",
            f"Longest error run spans rows {run['start_row']}:{run['end_row_exclusive']} with length {run['length']}.",
            "Open the error-atlas or sample-review queue around that row range.",
        )
    if not recs:
        add(
            30.0,
            "low",
            "monitoring",
            "Keep shadow replay in the evidence bundle",
            "Ordered windows did not show material degradation.",
            "Export the report so row-order stability evidence is retained with the model.",
        )

    recs.sort(key=lambda item: (-item["priority_score"], item["category"], item["title"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs[:6]


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))
