from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .experiments import compute_ece, fit_platt_scaling, probability_diagnostics
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_calibration_repair_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    calibration_fraction: float = 0.5,
    seed: int = 42,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compare post-hoc probability calibration repairs on a held-out evaluation split."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Calibration repair features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Calibration repair features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Calibration repair feature and label counts do not match.")
    if x.shape[0] < 2:
        raise ValueError("Calibration repair diagnostics need at least two samples.")
    if set(int(item) for item in np.unique(y)) - {0, 1}:
        raise ValueError("Calibration repair diagnostics require binary labels 0 or 1.")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    if probabilities.shape[0] != y.shape[0]:
        raise ValueError("Model returned a probability count that does not match the dataset.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)

    calibration_indices, evaluation_indices, split_source = _split_calibration_evaluation(
        y,
        calibration_fraction=float(calibration_fraction),
        seed=int(seed),
    )
    cal_y = y[calibration_indices]
    eval_y = y[evaluation_indices]
    cal_probs = probabilities[calibration_indices]
    eval_probs = probabilities[evaluation_indices]
    methods = [
        _method_report("raw", eval_y, eval_probs, n_bins=int(n_bins), params={}),
    ]

    platt_a, platt_b = fit_platt_scaling(cal_probs, cal_y)
    methods.append(
        _method_report(
            "platt",
            eval_y,
            _apply_platt(eval_probs, platt_a, platt_b),
            n_bins=int(n_bins),
            params={"a": float(platt_a), "b": float(platt_b)},
        )
    )

    isotonic_model = _fit_isotonic(cal_probs, cal_y)
    isotonic_probs = np.asarray(isotonic_model.predict(eval_probs), dtype=np.float32)
    methods.append(
        _method_report(
            "isotonic",
            eval_y,
            np.clip(isotonic_probs, 0.0, 1.0),
            n_bins=int(n_bins),
            params={
                "x_thresholds": [float(item) for item in isotonic_model.X_thresholds_],
                "y_thresholds": [float(item) for item in isotonic_model.y_thresholds_],
            },
        )
    )

    raw = methods[0]
    for method in methods[1:]:
        method["brier_improvement"] = float(raw["brier_score"] - method["brier_score"])
        method["ece_improvement"] = float(raw["ece"] - method["ece"])
        method["log_loss_improvement"] = float(raw["log_loss"] - method["log_loss"])
    best = min(methods, key=lambda item: (float(item["brier_score"]), float(item["ece"]), float(item["log_loss"])))
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "split": {
            "source": split_source,
            "calibration_fraction": float(calibration_fraction),
            "calibration_count": int(cal_y.shape[0]),
            "evaluation_count": int(eval_y.shape[0]),
            "seed": int(seed),
        },
        "methods": methods,
        "summary": {
            "recommended_method": best["method"],
            "recommended_brier_score": float(best["brier_score"]),
            "recommended_ece": float(best["ece"]),
            "recommended_log_loss": float(best["log_loss"]),
            "best_brier_improvement": float(raw["brier_score"] - best["brier_score"]),
            "best_ece_improvement": float(raw["ece"] - best["ece"]),
            "best_log_loss_improvement": float(raw["log_loss"] - best["log_loss"]),
            "warning": _warning(y, split_source, cal_y, eval_y, methods),
        },
    }


def format_calibration_repair_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Calibration repair: "
        f"best={summary.get('recommended_method') or '-'}, "
        f"Brier={float(summary.get('recommended_brier_score', 0.0)):.4f}, "
        f"ECE={float(summary.get('recommended_ece', 0.0)):.4f}, "
        f"dBrier={float(summary.get('best_brier_improvement', 0.0)):.4f}, "
        f"dECE={float(summary.get('best_ece_improvement', 0.0)):.4f}"
    )


def _method_report(
    method: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = probability_diagnostics(labels, probabilities, n_bins=n_bins)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    predictions = (probabilities >= 0.5).astype(np.int32)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)
    return {
        "method": method,
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))),
        "ece": float(compute_ece(labels, probabilities, n_bins=n_bins)),
        "max_calibration_error": float(diagnostics["max_calibration_error"]),
        "mean_probability": float(np.mean(probabilities)),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "calibration_bins": diagnostics.get("calibration_bins", []),
        "params": params,
        "brier_improvement": 0.0,
        "ece_improvement": 0.0,
        "log_loss_improvement": 0.0,
    }


def _apply_platt(probabilities: np.ndarray, a: float, b: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    return (1.0 / (1.0 + np.exp(-(float(a) * logits + float(b))))).astype(np.float32)


def _fit_isotonic(probabilities: np.ndarray, labels: np.ndarray):
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:  # pragma: no cover - requirements include scikit-learn.
        raise RuntimeError("scikit-learn is required for isotonic calibration repair diagnostics.") from exc

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(probabilities, labels)


def _split_calibration_evaluation(
    labels: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) == 2 and np.all(counts >= 2):
        calibration: list[int] = []
        evaluation: list[int] = []
        for label in unique:
            indices = np.where(labels == label)[0]
            rng.shuffle(indices)
            calibration_count = int(round(indices.shape[0] * calibration_fraction))
            calibration_count = max(1, min(indices.shape[0] - 1, calibration_count))
            calibration.extend(int(item) for item in indices[:calibration_count])
            evaluation.extend(int(item) for item in indices[calibration_count:])
        cal = np.asarray(calibration, dtype=np.int32)
        ev = np.asarray(evaluation, dtype=np.int32)
        rng.shuffle(cal)
        rng.shuffle(ev)
        return cal, ev, "posthoc_stratified_split"

    indices = np.arange(labels.shape[0], dtype=np.int32)
    rng.shuffle(indices)
    calibration_count = int(round(indices.shape[0] * calibration_fraction))
    calibration_count = max(1, min(indices.shape[0] - 1, calibration_count))
    return indices[:calibration_count], indices[calibration_count:], "posthoc_global_split"


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _warning(
    labels: np.ndarray,
    split_source: str,
    calibration_labels: np.ndarray,
    evaluation_labels: np.ndarray,
    methods: list[dict[str, Any]],
) -> str | None:
    messages: list[str] = []
    unique = set(int(item) for item in np.unique(labels))
    if unique == {0}:
        messages.append("All labels are negative; probability calibration evidence is limited.")
    elif unique == {1}:
        messages.append("All labels are positive; probability calibration evidence is limited.")
    if split_source == "posthoc_global_split":
        messages.append("Too few examples per class for a stratified calibration/evaluation split.")
    if len(set(int(item) for item in np.unique(calibration_labels))) < 2:
        messages.append("Calibration split has one observed class.")
    if len(set(int(item) for item in np.unique(evaluation_labels))) < 2:
        messages.append("Evaluation split has one observed class.")
    bin_counts = [
        int(bin_data.get("count", 0))
        for method in methods
        for bin_data in method.get("calibration_bins", [])
    ]
    if bin_counts and min(bin_counts) < 2:
        messages.append("Some reliability bins are small; calibration estimates may be noisy.")
    return " ".join(messages) if messages else None
