"""Learning-curve diagnostics: performance vs. training set size."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .experiments import train_single_model
from .modeling import ModelConfig


def learning_curve_points(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Train at increasing training fractions and return validation F1 / accuracy."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 8:
        raise ValueError("Learning curves need at least 8 samples.")

    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    holdout_count = max(2, int(round(x.shape[0] * 0.2)))
    val_idx = indices[:holdout_count]
    train_pool = indices[holdout_count:]
    x_val, y_val = x[val_idx], y[val_idx]

    points: list[dict[str, Any]] = []
    for fraction in fractions:
        frac = min(max(float(fraction), 0.1), 1.0)
        n_train = max(4, int(round(len(train_pool) * frac)))
        n_train = min(n_train, len(train_pool))
        train_idx = train_pool[:n_train]
        subset_x = x[train_idx]
        subset_y = y[train_idx]
        merged_x = np.concatenate([subset_x, x_val], axis=0)
        merged_y = np.concatenate([subset_y, y_val], axis=0)
        result = train_single_model(merged_x, merged_y, config)
        points.append(
            {
                "train_fraction": frac,
                "train_samples": int(n_train),
                "f1": float(result.metrics.get("f1", 0.0)),
                "accuracy": float(result.metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(result.metrics.get("balanced_accuracy", 0.0)),
                "validation_loss": float(result.metrics.get("validation_loss", 0.0)),
            }
        )
    return points


def run_learning_curve_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Return a durable learning-curve report for reports and sidecars."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    points = learning_curve_points(x, y, config, fractions=fractions, seed=seed)
    summary = _summary(points)
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]) if x.ndim == 2 else None,
        "fractions": [float(point["train_fraction"]) for point in points],
        "dataset_fingerprint": learning_curve_dataset_fingerprint(x, y),
        "config": config.to_dict(),
        "summary": summary,
        "points": points,
    }


def format_learning_curve_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Learning curve: "
        f"verdict={summary.get('verdict', '-')}, "
        f"best_F1={float(summary.get('best_f1', 0.0)):.4f}, "
        f"final_F1={float(summary.get('final_f1', 0.0)):.4f}, "
        f"gain={float(summary.get('f1_gain', 0.0)):.4f}, "
        f"next={summary.get('recommended_next_step') or 'none'}"
    )


def learning_curve_dataset_fingerprint(features: np.ndarray, labels: np.ndarray) -> str:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    payload = np.ascontiguousarray(np.column_stack([x, y])).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {
            "verdict": "no_points",
            "best_f1": 0.0,
            "final_f1": 0.0,
            "f1_gain": 0.0,
            "recommended_next_step": "Run learning curve after adding enough labeled data.",
        }
    first = points[0]
    final = points[-1]
    best = max(points, key=lambda item: (float(item.get("f1", 0.0)), float(item.get("accuracy", 0.0))))
    first_f1 = float(first.get("f1", 0.0))
    final_f1 = float(final.get("f1", 0.0))
    best_f1 = float(best.get("f1", 0.0))
    gain = final_f1 - first_f1
    best_gap = best_f1 - final_f1
    if final_f1 < 0.60:
        verdict = "underfit_or_noisy"
        next_step = "Improve features, labels, or model search before relying on this configuration."
    elif gain >= 0.08:
        verdict = "more_data_helpful"
        next_step = "Collect more labeled rows; validation F1 is still improving with training size."
    elif best_gap >= 0.06:
        verdict = "capacity_or_regularization_review"
        next_step = "Review regularization, feature selection, and threshold stability; later fractions did not dominate."
    else:
        verdict = "stable_enough"
        next_step = "Use external holdout or robustness diagnostics before promotion."
    return {
        "verdict": verdict,
        "first_f1": first_f1,
        "final_f1": final_f1,
        "best_f1": best_f1,
        "best_fraction": float(best.get("train_fraction", 0.0)),
        "f1_gain": float(gain),
        "best_gap_vs_final": float(best_gap),
        "recommended_next_step": next_step,
    }
