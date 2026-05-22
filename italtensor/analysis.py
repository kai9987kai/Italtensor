from __future__ import annotations

from typing import Any
import numpy as np

from .experiments import _classification_metrics_at_threshold


def _get_flat_weights(model: Any) -> np.ndarray:
    """Helper to extract and flatten trainable weights from Numpy or Keras models."""
    if hasattr(model, "trainable_variables"):
        # Keras model
        vars_list = [var.numpy().flatten() for var in model.trainable_variables]
        if not vars_list:
            return np.array([], dtype=np.float32)
        return np.concatenate(vars_list)
    elif hasattr(model, "weights"):
        # NumpyBinaryClassifier
        weights = np.asarray(model.weights, dtype=np.float32).flatten()
        # Include bias if available to match all model parameter statistics
        if hasattr(model, "bias"):
            bias = np.array([model.bias], dtype=np.float32)
            return np.concatenate([weights, bias])
        return weights
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def weight_statistics(model: Any) -> dict[str, float]:
    """Compute mean, std, min, max, sparsity (% zeros), L1/L2 norms of model weights."""
    flat_weights = _get_flat_weights(model)
    if flat_weights.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sparsity": 100.0,
            "l1_norm": 0.0,
            "l2_norm": 0.0,
        }

    mean = float(np.mean(flat_weights))
    std = float(np.std(flat_weights))
    minimum = float(np.min(flat_weights))
    maximum = float(np.max(flat_weights))
    sparsity = float(np.mean(flat_weights == 0.0) * 100.0)
    l1_norm = float(np.sum(np.abs(flat_weights)))
    l2_norm = float(np.sqrt(np.sum(flat_weights ** 2)))

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "sparsity": sparsity,
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
    }


def model_similarity(model_a: Any, model_b: Any) -> float:
    """Cosine similarity between two models' weight vectors."""
    w_a = _get_flat_weights(model_a)
    w_b = _get_flat_weights(model_b)

    if w_a.size != w_b.size:
        raise ValueError(
            f"Cannot compute similarity: models have different weight sizes "
            f"({w_a.size} vs {w_b.size})."
        )

    if w_a.size == 0:
        return 0.0

    norm_a = np.sqrt(np.sum(w_a ** 2))
    norm_b = np.sqrt(np.sum(w_b ** 2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = np.sum(w_a * w_b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return float(cosine_sim)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    B: int = 1000,
    alpha: float = 0.05,
    threshold: float = 0.5,
    random_seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Resample validation predictions B times, compute 95% CI for F1, accuracy, balanced accuracy."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_pred_probs = np.asarray(y_pred_probs, dtype=np.float32).reshape(-1)

    if y_true.size != y_pred_probs.size:
        raise ValueError("y_true and y_pred_probs must have the same length.")

    n_samples = y_true.size
    if n_samples == 0:
        return {
            "f1": (0.0, 0.0),
            "accuracy": (0.0, 0.0),
            "balanced_accuracy": (0.0, 0.0),
        }

    rng = np.random.default_rng(random_seed)

    f1_scores = []
    accuracy_scores = []
    balanced_accuracy_scores = []

    for _ in range(B):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_b = y_true[indices]
        y_prob_b = y_pred_probs[indices]

        metrics = _classification_metrics_at_threshold(y_true_b, y_prob_b, threshold)
        f1_scores.append(metrics["f1"])
        accuracy_scores.append(metrics["accuracy"])
        balanced_accuracy_scores.append(metrics["balanced_accuracy"])

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    return {
        "f1": (
            float(np.percentile(f1_scores, lower_pct)),
            float(np.percentile(f1_scores, upper_pct)),
        ),
        "accuracy": (
            float(np.percentile(accuracy_scores, lower_pct)),
            float(np.percentile(accuracy_scores, upper_pct)),
        ),
        "balanced_accuracy": (
            float(np.percentile(balanced_accuracy_scores, lower_pct)),
            float(np.percentile(balanced_accuracy_scores, upper_pct)),
        ),
    }


def compute_weight_histogram(model: Any, bins: int = 10) -> dict[str, list[float]]:
    """Bin weights into histogram for distribution analysis."""
    flat_weights = _get_flat_weights(model)
    if flat_weights.size == 0:
        return {
            "counts": [0.0] * bins,
            "bin_edges": [0.0] * (bins + 1),
        }

    counts, bin_edges = np.histogram(flat_weights, bins=bins)
    return {
        "counts": [float(c) for c in counts],
        "bin_edges": [float(e) for e in bin_edges],
    }
