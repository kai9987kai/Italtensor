"""Dataset cartography: confidence vs. local prediction variability.

Inspired by Swayamdipta et al., EMNLP 2020 (Dataset Cartography): map examples into
easy-to-learn, ambiguous, hard, and overconfident-wrong regions using model behavior.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def _neighborhood_variability(
    model: Any,
    features: np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None,
    n_perturbations: int = 5,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Estimate local variability as std of predictions under small feature jitter."""
    rng = np.random.default_rng(seed)
    prepared = preprocessor.transform(features) if preprocessor is not None else features
    base = predict_probability(model, prepared).reshape(-1)
    if n_perturbations <= 0:
        return np.zeros_like(base)
    stacked = [base]
    col_std = np.std(features, axis=0)
    scale = np.where(col_std > 1e-8, col_std * noise_scale, noise_scale)
    for _ in range(n_perturbations):
        noise = rng.normal(0.0, 1.0, size=features.shape).astype(np.float32) * scale
        perturbed = features + noise
        prep = preprocessor.transform(perturbed) if preprocessor is not None else perturbed
        stacked.append(predict_probability(model, prep).reshape(-1))
    return np.std(np.stack(stacked, axis=0), axis=0).astype(np.float32)


def run_dataset_cartography(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    max_items_per_region: int = 8,
    n_perturbations: int = 5,
) -> dict[str, Any]:
    """Classify rows into cartography regions for curriculum and audit workflows."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Cartography features and labels must align on a 2D feature matrix.")
    if x.shape[0] == 0:
        raise ValueError("Cartography needs at least one sample.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared).reshape(-1)
    predicted = (probabilities >= threshold).astype(np.int32)
    confidence = np.where(y == 1, probabilities, 1.0 - probabilities)
    variability = _neighborhood_variability(
        model,
        x,
        preprocessor=preprocessor,
        n_perturbations=n_perturbations,
    )

    conf_med = float(np.median(confidence))
    var_med = float(np.median(variability))
    regions: dict[str, list[dict[str, Any]]] = {
        "easy_to_learn": [],
        "ambiguous": [],
        "hard_to_learn": [],
        "overconfident_wrong": [],
    }

    for index in range(x.shape[0]):
        item = {
            "row_index": int(index),
            "label": int(y[index]),
            "predicted_label": int(predicted[index]),
            "probability": float(probabilities[index]),
            "confidence": float(confidence[index]),
            "variability": float(variability[index]),
            "margin": float(abs(probabilities[index] - threshold)),
        }
        disagrees = predicted[index] != y[index]
        high_conf = confidence[index] >= conf_med
        high_var = variability[index] >= var_med
        if disagrees and confidence[index] >= conf_med:
            regions["overconfident_wrong"].append(item)
        elif high_conf and not high_var:
            regions["easy_to_learn"].append(item)
        elif not high_conf and high_var:
            regions["ambiguous"].append(item)
        elif not high_conf and not high_var:
            regions["hard_to_learn"].append(item)
        else:
            regions["easy_to_learn"].append(item)

    for key in regions:
        if key == "overconfident_wrong":
            regions[key].sort(key=lambda row: (-row["confidence"], row["row_index"]))
        elif key == "ambiguous":
            regions[key].sort(key=lambda row: (-row["variability"], row["confidence"], row["row_index"]))
        elif key == "hard_to_learn":
            regions[key].sort(key=lambda row: (row["confidence"], row["variability"], row["row_index"]))
        else:
            regions[key].sort(key=lambda row: (-row["confidence"], row["variability"], row["row_index"]))
        regions[key] = regions[key][: max(1, int(max_items_per_region))]

    counts = {name: int(np.sum(_region_mask(name, confidence, variability, predicted, y, conf_med, var_med))) for name in (
        "easy_to_learn",
        "ambiguous",
        "hard_to_learn",
        "overconfident_wrong",
    )}

    return {
        "sample_count": int(x.shape[0]),
        "threshold": float(threshold),
        "median_confidence": conf_med,
        "median_variability": var_med,
        "region_counts": counts,
        "regions": regions,
    }


def _region_mask(
    name: str,
    confidence: np.ndarray,
    variability: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray,
    conf_med: float,
    var_med: float,
) -> np.ndarray:
    disagrees = predicted != labels
    high_conf = confidence >= conf_med
    high_var = variability >= var_med
    if name == "overconfident_wrong":
        return disagrees & high_conf
    if name == "ambiguous":
        return (~high_conf) & high_var
    if name == "hard_to_learn":
        return (~high_conf) & (~high_var)
    return high_conf & (~disagrees)


def format_cartography_summary(report: dict[str, Any]) -> str:
    counts = report.get("region_counts", {})
    return (
        "Dataset cartography: "
        f"easy={int(counts.get('easy_to_learn', 0))}, "
        f"ambiguous={int(counts.get('ambiguous', 0))}, "
        f"hard={int(counts.get('hard_to_learn', 0))}, "
        f"overconfident_wrong={int(counts.get('overconfident_wrong', 0))} "
        f"(median conf={float(report.get('median_confidence', 0)):.3f}, "
        f"median var={float(report.get('median_variability', 0)):.3f})"
    )
