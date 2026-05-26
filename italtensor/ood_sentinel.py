from __future__ import annotations

from typing import Any, Sequence

import numpy as np


EPSILON = 1e-12
ROBUST_NORMAL_SCALE = 1.4826
PREVIEW_COLUMNS = 8


def run_ood_sentinel(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: Any | None = None,
    threshold: float = 0.5,
    max_rows: int = 12,
    min_reference_samples: int = 4,
) -> dict[str, Any]:
    """Rank rows that look geometrically isolated or surprising to a binary model."""
    x, y = _validate_inputs(features, labels)
    threshold = _validate_probability_threshold(threshold)
    max_rows = _validate_positive_int(max_rows, "max_rows")
    min_reference_samples = _validate_positive_int(min_reference_samples, "min_reference_samples")
    if min_reference_samples < 2:
        raise ValueError("OOD sentinel min_reference_samples must be at least two.")
    if x.shape[0] < min_reference_samples:
        raise ValueError("OOD sentinel needs at least min_reference_samples rows.")

    robust = _robust_geometry(x)
    probabilities = None
    model_metrics: dict[str, np.ndarray] | None = None
    if model is not None:
        prepared = _prepare_for_model(preprocessor, x)
        probabilities = _predict_probabilities(model, prepared, expected_rows=x.shape[0])
        model_metrics = _model_row_metrics(probabilities, y, threshold)

    all_rows = _rank_rows(
        x=x,
        y=y,
        robust=robust,
        threshold=threshold,
        probabilities=probabilities,
        model_metrics=model_metrics,
    )
    returned_rows = all_rows[:max_rows]
    summary = _summary(
        all_rows,
        robust,
        model_metrics=model_metrics,
        probabilities=probabilities,
        threshold=threshold,
    )
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "model_used": bool(model is not None),
        "summary": summary,
        "rows": returned_rows,
    }


def format_ood_sentinel_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_row = summary.get("top_row_index")
    top_text = "-" if top_row is None else str(int(top_row))
    model_text = "on" if bool(report.get("model_used", False)) else "off"
    pieces = [
        "OOD sentinel: ",
        f"top_row={top_text}, ",
        f"max_score={float(summary.get('max_ood_score', 0.0)):.4f}, ",
        f"flagged={int(summary.get('flagged_count', 0))}, ",
        f"max_z={float(summary.get('max_abs_robust_z', 0.0)):.4f}, ",
        f"max_nn={float(summary.get('max_nearest_neighbor_distance', 0.0)):.4f}, ",
        f"model={model_text}",
    ]
    if bool(report.get("model_used", False)):
        pieces.append(f", max_loss={float(summary.get('max_model_loss', 0.0)):.4f}")
    return "".join(pieces)


def _validate_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOD sentinel features must be numeric.") from exc
    if x.ndim != 2:
        raise ValueError("OOD sentinel features must be a 2D array.")
    if x.shape[0] == 0:
        raise ValueError("OOD sentinel needs at least one sample.")
    if x.shape[1] == 0:
        raise ValueError("OOD sentinel needs at least one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError("OOD sentinel features must be finite numbers.")

    raw_labels = np.asarray(labels)
    if raw_labels.ndim == 0 or raw_labels.ndim > 2:
        raise ValueError("OOD sentinel labels must be a flat binary array.")
    if raw_labels.ndim == 2 and 1 not in raw_labels.shape:
        raise ValueError("OOD sentinel labels must be a flat binary array.")
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOD sentinel labels must be numeric.") from exc
    if y.shape[0] != x.shape[0]:
        raise ValueError("OOD sentinel feature and label counts do not match.")
    if not np.all(np.isfinite(y)):
        raise ValueError("OOD sentinel labels must be finite numbers.")
    if set(np.unique(y).tolist()) - {0.0, 1.0}:
        raise ValueError("OOD sentinel requires binary labels 0 or 1.")
    return x, y.astype(np.int32)


def _validate_probability_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError("OOD sentinel threshold must be between 0 and 1.")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"OOD sentinel {name} must be positive.")
    return parsed


def _robust_geometry(x: np.ndarray) -> dict[str, np.ndarray | int]:
    medians = np.median(x, axis=0)
    absolute_deviations = np.abs(x - medians)
    mad = np.median(absolute_deviations, axis=0)
    zero_mad_count = int(np.sum(mad <= EPSILON))
    robust_scale = np.where(mad > EPSILON, ROBUST_NORMAL_SCALE * mad, 1.0)
    robust_z = (x - medians) / robust_scale
    abs_robust_z = np.abs(robust_z)
    nearest_neighbor_distance = _nearest_neighbor_distances(robust_z)
    nearest_center = float(np.median(nearest_neighbor_distance))
    nearest_mad = float(np.median(np.abs(nearest_neighbor_distance - nearest_center)))
    nearest_scale = ROBUST_NORMAL_SCALE * nearest_mad if nearest_mad > EPSILON else max(nearest_center, 1.0)
    nearest_excess = np.maximum(0.0, (nearest_neighbor_distance - nearest_center) / nearest_scale)
    return {
        "medians": medians,
        "mad": mad,
        "robust_scale": robust_scale,
        "robust_z": robust_z,
        "max_abs_robust_z": np.max(abs_robust_z, axis=1),
        "mean_abs_robust_z": np.mean(abs_robust_z, axis=1),
        "nearest_neighbor_distance": nearest_neighbor_distance,
        "nearest_neighbor_excess": nearest_excess,
        "zero_mad_feature_count": zero_mad_count,
    }


def _nearest_neighbor_distances(robust_z: np.ndarray) -> np.ndarray:
    distances = np.empty(robust_z.shape[0], dtype=np.float64)
    for index in range(robust_z.shape[0]):
        delta = robust_z - robust_z[index]
        row_distances = np.sqrt(np.sum(delta * delta, axis=1))
        row_distances[index] = np.inf
        distances[index] = float(np.min(row_distances))
    return distances


def _prepare_for_model(preprocessor: Any | None, x: np.ndarray) -> np.ndarray:
    if preprocessor is None:
        return x
    if hasattr(preprocessor, "transform"):
        prepared = preprocessor.transform(x)
    elif callable(preprocessor):
        prepared = preprocessor(x)
    else:
        raise ValueError("OOD sentinel preprocessor must be callable or expose transform().")
    try:
        prepared_array = np.asarray(prepared, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOD sentinel preprocessor output must be numeric.") from exc
    if prepared_array.ndim != 2:
        raise ValueError("OOD sentinel preprocessor output must be a 2D array.")
    if prepared_array.shape[0] != x.shape[0]:
        raise ValueError("OOD sentinel preprocessor output row count does not match features.")
    if prepared_array.shape[1] == 0:
        raise ValueError("OOD sentinel preprocessor output needs at least one feature.")
    if not np.all(np.isfinite(prepared_array)):
        raise ValueError("OOD sentinel preprocessor output must be finite numbers.")
    return prepared_array


def _predict_probabilities(model: Any, prepared: np.ndarray, *, expected_rows: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = _call_model_method(model.predict_proba, prepared)
    elif hasattr(model, "predict"):
        raw = _call_model_method(model.predict, prepared)
    elif callable(model):
        raw = model(prepared)
    else:
        raise ValueError("OOD sentinel model must be callable or expose predict()/predict_proba().")

    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    probabilities = np.asarray(raw, dtype=np.float64)
    if probabilities.ndim == 0 and expected_rows == 1:
        probabilities = probabilities.reshape(1)
    if probabilities.ndim == 2:
        if probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
        elif probabilities.shape[1] == 1:
            probabilities = probabilities[:, 0]
        else:
            raise ValueError("OOD sentinel model probabilities must have one or two columns.")
    elif probabilities.ndim != 1:
        raise ValueError("OOD sentinel model probabilities must be a flat array.")
    probabilities = probabilities.reshape(-1)
    if probabilities.shape[0] != expected_rows:
        raise ValueError("OOD sentinel model probability count does not match features.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("OOD sentinel model probabilities must be finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("OOD sentinel model probabilities must be between 0 and 1.")
    return probabilities


def _call_model_method(method: Any, prepared: np.ndarray) -> Any:
    try:
        return method(prepared, verbose=0)
    except TypeError:
        return method(prepared)


def _model_row_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    losses = -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))
    predicted = (probabilities >= threshold).astype(np.int32)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    uncertainty = 1.0 - np.clip((confidence - 0.5) / 0.5, 0.0, 1.0)
    misclassified = predicted != labels
    loss_component = losses / (losses + 1.0)
    model_score = 0.50 * loss_component + 0.20 * uncertainty + 0.30 * misclassified.astype(np.float64)
    return {
        "loss": losses,
        "predicted": predicted,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "misclassified": misclassified,
        "model_score": model_score,
    }


def _rank_rows(
    *,
    x: np.ndarray,
    y: np.ndarray,
    robust: dict[str, np.ndarray | int],
    threshold: float,
    probabilities: np.ndarray | None,
    model_metrics: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    max_abs_z = np.asarray(robust["max_abs_robust_z"], dtype=np.float64)
    mean_abs_z = np.asarray(robust["mean_abs_robust_z"], dtype=np.float64)
    nearest = np.asarray(robust["nearest_neighbor_distance"], dtype=np.float64)
    nearest_excess = np.asarray(robust["nearest_neighbor_excess"], dtype=np.float64)
    robust_z = np.asarray(robust["robust_z"], dtype=np.float64)
    geometry_score = (
        0.55 * _bounded_component(max_abs_z / 3.0)
        + 0.25 * _bounded_component(mean_abs_z / 2.0)
        + 0.20 * _bounded_component(nearest_excess)
    )
    if model_metrics is None:
        model_score = np.zeros(x.shape[0], dtype=np.float64)
    else:
        model_score = np.asarray(model_metrics["model_score"], dtype=np.float64)
    ood_score = geometry_score + model_score

    rows: list[dict[str, Any]] = []
    for index in range(x.shape[0]):
        z_row = robust_z[index]
        top_feature = int(np.argmax(np.abs(z_row)))
        row = {
            "row_index": int(index),
            "label": int(y[index]),
            "ood_score": float(ood_score[index]),
            "geometry_score": float(geometry_score[index]),
            "model_score": float(model_score[index]),
            "max_abs_robust_z": float(max_abs_z[index]),
            "mean_abs_robust_z": float(mean_abs_z[index]),
            "nearest_neighbor_distance": float(nearest[index]),
            "top_feature_index": top_feature,
            "top_feature_robust_z": float(z_row[top_feature]),
            "feature_preview": [float(value) for value in x[index, :PREVIEW_COLUMNS]],
            "robust_z_preview": [float(value) for value in z_row[:PREVIEW_COLUMNS]],
            "risk_flags": _risk_flags(
                ood_score=float(ood_score[index]),
                threshold=threshold,
                max_abs_z=float(max_abs_z[index]),
                mean_abs_z=float(mean_abs_z[index]),
                nearest_excess=float(nearest_excess[index]),
                model_metrics=model_metrics,
                index=index,
            ),
        }
        if probabilities is not None and model_metrics is not None:
            row.update(
                {
                    "probability": float(probabilities[index]),
                    "predicted_label": int(model_metrics["predicted"][index]),
                    "model_confidence": float(model_metrics["confidence"][index]),
                    "model_loss": float(model_metrics["loss"][index]),
                    "loss": float(model_metrics["loss"][index]),
                    "misclassified": bool(model_metrics["misclassified"][index]),
                }
            )
        else:
            row.update(
                {
                    "probability": None,
                    "predicted_label": None,
                    "model_confidence": None,
                    "model_loss": None,
                    "loss": None,
                    "misclassified": None,
                }
            )
        rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(item["ood_score"]),
            -float(item["max_abs_robust_z"]),
            -float(item["nearest_neighbor_distance"]),
            int(item["row_index"]),
        )
    )
    return rows


def _bounded_component(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return clipped / (1.0 + clipped)


def _risk_flags(
    *,
    ood_score: float,
    threshold: float,
    max_abs_z: float,
    mean_abs_z: float,
    nearest_excess: float,
    model_metrics: dict[str, np.ndarray] | None,
    index: int,
) -> list[str]:
    flags: list[str] = []
    if ood_score >= threshold:
        flags.append("score_above_threshold")
    if max_abs_z >= 4.0:
        flags.append("feature_tail")
    if mean_abs_z >= 2.0:
        flags.append("broad_feature_tail")
    if nearest_excess >= 3.0:
        flags.append("isolated_neighbor")
    if model_metrics is not None:
        if bool(model_metrics["misclassified"][index]):
            flags.append("misclassified")
        if float(model_metrics["loss"][index]) >= 1.0:
            flags.append("high_model_loss")
        if float(model_metrics["confidence"][index]) <= 0.60:
            flags.append("low_model_confidence")
    return flags


def _summary(
    rows: list[dict[str, Any]],
    robust: dict[str, np.ndarray | int],
    *,
    model_metrics: dict[str, np.ndarray] | None,
    probabilities: np.ndarray | None,
    threshold: float,
) -> dict[str, Any]:
    scores = np.asarray([float(row["ood_score"]) for row in rows], dtype=np.float64)
    max_abs_z = np.asarray(robust["max_abs_robust_z"], dtype=np.float64)
    mean_abs_z = np.asarray(robust["mean_abs_robust_z"], dtype=np.float64)
    nearest = np.asarray(robust["nearest_neighbor_distance"], dtype=np.float64)
    top = rows[0] if rows else {}
    warnings: list[str] = []
    zero_mad_count = int(robust["zero_mad_feature_count"])
    if zero_mad_count:
        warnings.append(f"{zero_mad_count} zero-MAD feature(s) used unit robust scale")
    if probabilities is None:
        max_model_loss = 0.0
        mean_model_loss = 0.0
        misclassification_count = 0
    else:
        losses = np.asarray(model_metrics["loss"], dtype=np.float64) if model_metrics is not None else np.zeros(0)
        max_model_loss = float(np.max(losses)) if losses.size else 0.0
        mean_model_loss = float(np.mean(losses)) if losses.size else 0.0
        misclassification_count = int(np.sum(model_metrics["misclassified"])) if model_metrics is not None else 0
    flagged_count = int(np.sum(scores >= threshold)) if scores.size else 0
    return {
        "top_row_index": top.get("row_index"),
        "max_ood_score": float(np.max(scores)) if scores.size else 0.0,
        "mean_ood_score": float(np.mean(scores)) if scores.size else 0.0,
        "flagged_count": flagged_count,
        "flagged_row_count": flagged_count,
        "max_abs_robust_z": float(np.max(max_abs_z)) if max_abs_z.size else 0.0,
        "mean_abs_robust_z": float(np.mean(mean_abs_z)) if mean_abs_z.size else 0.0,
        "max_nearest_neighbor_distance": float(np.max(nearest)) if nearest.size else 0.0,
        "mean_nearest_neighbor_distance": float(np.mean(nearest)) if nearest.size else 0.0,
        "model_loss_available": bool(probabilities is not None),
        "max_model_loss": max_model_loss,
        "mean_model_loss": mean_model_loss,
        "misclassification_count": misclassification_count,
        "warning": "; ".join(warnings) if warnings else None,
    }
