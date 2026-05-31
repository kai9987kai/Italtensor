from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .experiments import evaluate_predictions
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_threshold_stability(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    current_threshold: float = 0.5,
    bootstrap_samples: int = 80,
    resamples: int | None = None,
    seed: int = 42,
    grid_size: int = 101,
) -> dict[str, Any]:
    """Bootstrap the active model's threshold choice on the loaded labeled rows."""
    x, y = _validate_inputs(features, labels)
    current_threshold = _validate_threshold(current_threshold)
    bootstrap_samples = max(8, int(resamples if resamples is not None else bootstrap_samples))
    grid_size = max(5, int(grid_size))
    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Threshold stability preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    thresholds = _threshold_grid(probabilities, current_threshold, grid_size)
    full_current = _threshold_point(y, probabilities, current_threshold)
    full_best = _best_threshold_point(y, probabilities, thresholds, current_threshold)
    rng = np.random.default_rng(int(seed))
    sample_indices = _stratified_bootstrap_indices(y, rng, bootstrap_samples)
    runs = [
        _bootstrap_run(index, indices, y, probabilities, thresholds, current_threshold)
        for index, indices in enumerate(sample_indices)
    ]
    summary = _summary(runs, full_current, full_best, current_threshold)
    recommendations = _recommendations(summary)
    summary["recommendation"] = recommendations[0]["action"] if recommendations else None
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "current_threshold": float(current_threshold),
        "bootstrap_samples": int(bootstrap_samples),
        "resamples": int(bootstrap_samples),
        "seed": int(seed),
        "grid_size": int(grid_size),
        "dataset_fingerprint": threshold_stability_dataset_fingerprint(x, y),
        "full_dataset": {
            "current": full_current,
            "best_f1": full_best,
        },
        "summary": summary,
        "threshold_interval": summary["threshold_interval"],
        "resample_runs": _compact_runs(runs),
        "recommendations": recommendations,
    }


def format_threshold_stability_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    interval = summary.get("threshold_interval", {})
    return (
        "Threshold stability: "
        f"verdict={summary.get('verdict', '-')}, "
        f"current={float(report.get('current_threshold', 0.5)):.4f}, "
        f"median={float(summary.get('median_best_threshold', 0.5)):.4f}, "
        f"q05={float(interval.get('q05', 0.0)):.4f}, "
        f"q95={float(interval.get('q95', 1.0)):.4f}, "
        f"spread={float(summary.get('threshold_spread', 0.0)):.4f}, "
        f"gain={float(summary.get('median_f1_gain_vs_current', 0.0)):.4f}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def threshold_stability_dataset_fingerprint(
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
        raise ValueError("Threshold stability features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Threshold stability features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Threshold stability feature and label counts do not match.")
    if x.shape[0] < 6:
        raise ValueError("Threshold stability needs at least six samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Threshold stability features must be finite numbers.")
    if np.unique(y).size < 2:
        raise ValueError("Threshold stability needs both classes present.")
    return x, y


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold stability labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Threshold stability labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Threshold stability labels must be binary 0/1.")
    return y.astype(np.int32)


def _validate_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("current_threshold must be between 0 and 1.")
    return threshold


def _threshold_grid(probabilities: np.ndarray, current_threshold: float, grid_size: int) -> np.ndarray:
    values = np.concatenate(
        [
            np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64),
            probabilities.astype(np.float64),
            np.asarray([current_threshold], dtype=np.float64),
        ]
    )
    return np.unique(np.clip(values, 0.0, 1.0))


def _threshold_point(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = evaluate_predictions(labels, probabilities, float(threshold))
    return {
        "threshold": float(threshold),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "true_positive": int(metrics["true_positive"]),
        "true_negative": int(metrics["true_negative"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
    }


def _best_threshold_point(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    current_threshold: float,
) -> dict[str, float | int]:
    points = [_threshold_point(labels, probabilities, float(threshold)) for threshold in thresholds]
    return max(
        points,
        key=lambda point: (
            float(point["f1"]),
            float(point["balanced_accuracy"]),
            float(point["accuracy"]),
            -abs(float(point["threshold"]) - current_threshold),
        ),
    )


def _stratified_bootstrap_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> list[np.ndarray]:
    indices_by_class = [np.where(labels == value)[0] for value in (0, 1)]
    runs: list[np.ndarray] = []
    for _ in range(resamples):
        parts = [
            rng.choice(indices, size=indices.shape[0], replace=True)
            for indices in indices_by_class
        ]
        combined = np.concatenate(parts).astype(np.int32)
        rng.shuffle(combined)
        runs.append(combined)
    return runs


def _bootstrap_run(
    run_index: int,
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    current_threshold: float,
) -> dict[str, Any]:
    sample_labels = labels[indices]
    sample_probabilities = probabilities[indices]
    current = _threshold_point(sample_labels, sample_probabilities, current_threshold)
    best = _best_threshold_point(sample_labels, sample_probabilities, thresholds, current_threshold)
    return {
        "run_index": int(run_index),
        "best_threshold": float(best["threshold"]),
        "best_f1": float(best["f1"]),
        "current_f1": float(current["f1"]),
        "f1_gain_vs_current": float(best["f1"] - current["f1"]),
        "best_balanced_accuracy": float(best["balanced_accuracy"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
    }


def _summary(
    runs: list[dict[str, Any]],
    full_current: dict[str, float | int],
    full_best: dict[str, float | int],
    current_threshold: float,
) -> dict[str, Any]:
    thresholds = np.asarray([float(run["best_threshold"]) for run in runs], dtype=np.float64)
    gains = np.asarray([float(run["f1_gain_vs_current"]) for run in runs], dtype=np.float64)
    best_f1 = np.asarray([float(run["best_f1"]) for run in runs], dtype=np.float64)
    q05, q25, q50, q75, q95 = np.quantile(thresholds, [0.05, 0.25, 0.5, 0.75, 0.95])
    threshold_spread = float(q95 - q05)
    iqr = float(q75 - q25)
    close_to_full = np.abs(thresholds - float(full_best["threshold"])) <= 0.05
    current_inside_interval = bool(q05 <= current_threshold <= q95)
    median_gain = float(np.median(gains))
    p90_gain = float(np.quantile(gains, 0.90))
    verdict = _verdict(
        threshold_spread=threshold_spread,
        iqr=iqr,
        median_gain=median_gain,
        current_inside_interval=current_inside_interval,
    )
    return {
        "verdict": verdict,
        "current_threshold": float(current_threshold),
        "full_best_threshold": float(full_best["threshold"]),
        "full_current_f1": float(full_current["f1"]),
        "full_best_f1": float(full_best["f1"]),
        "full_f1_gain_vs_current": float(float(full_best["f1"]) - float(full_current["f1"])),
        "median_best_threshold": float(q50),
        "threshold_interval": {"q05": float(q05), "q25": float(q25), "q50": float(q50), "q75": float(q75), "q95": float(q95)},
        "threshold_spread": threshold_spread,
        "threshold_iqr": iqr,
        "threshold_std": float(np.std(thresholds)),
        "current_inside_interval": current_inside_interval,
        "selection_agreement_rate": float(np.mean(close_to_full)),
        "median_best_f1": float(np.median(best_f1)),
        "median_f1_gain_vs_current": median_gain,
        "p90_f1_gain_vs_current": p90_gain,
        "positive_gain_rate": float(np.mean(gains > 0.01)),
        "completed_bootstrap_count": int(len(runs)),
        "skipped_bootstrap_count": 0,
    }


def _verdict(
    *,
    threshold_spread: float,
    iqr: float,
    median_gain: float,
    current_inside_interval: bool,
) -> str:
    if threshold_spread >= 0.35 or (not current_inside_interval and median_gain >= 0.08):
        return "unstable_threshold"
    if threshold_spread >= 0.18 or iqr >= 0.10 or not current_inside_interval or median_gain >= 0.04:
        return "threshold_stability_review"
    return "stable_threshold"


def _recommendations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = str(summary.get("verdict", "stable_threshold"))
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

    if verdict == "unstable_threshold":
        add(
            88.0,
            "high",
            "threshold",
            "Threshold choice is unstable",
            f"Bootstrap 5-95% threshold spread is {summary['threshold_spread']:.3f}.",
            "Collect more validation rows or choose a conservative threshold range before promotion.",
        )
    elif verdict == "threshold_stability_review":
        add(
            64.0,
            "medium",
            "threshold",
            "Review threshold stability",
            f"Median bootstrap F1 gain vs current is {summary['median_f1_gain_vs_current']:.3f}.",
            "Compare Threshold tradeoff, Decision curve, and this stability interval before locking the threshold.",
        )
    if not bool(summary.get("current_inside_interval", False)):
        add(
            58.0,
            "medium",
            "threshold",
            "Current threshold sits outside the bootstrap interval",
            "The selected threshold is not inside the bootstrap 5-95% best-threshold interval.",
            "Revisit the active threshold or document why this operating point is intentionally off the empirical optimum.",
        )
    if not recs:
        add(
            30.0,
            "low",
            "promotion",
            "Keep threshold stability evidence with the model",
            "Bootstrap threshold spread is narrow and the current threshold is inside the interval.",
            "Export the report or model sidecar so operating-point stability evidence is retained.",
        )
    recs.sort(key=lambda item: (-item["priority_score"], item["category"], item["title"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs[:6]


def _compact_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(runs) <= 30:
        return runs
    selected = set(np.linspace(0, len(runs) - 1, 20, dtype=np.int32).tolist())
    selected.update(
        sorted(
            range(len(runs)),
            key=lambda index: (-float(runs[index]["f1_gain_vs_current"]), float(runs[index]["best_threshold"])),
        )[:5]
    )
    selected.update(
        sorted(
            range(len(runs)),
            key=lambda index: (float(runs[index]["f1_gain_vs_current"]), float(runs[index]["best_threshold"])),
        )[:5]
    )
    return [runs[index] for index in sorted(selected)]
