from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .experiments import probability_diagnostics
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_reliability_atlas(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    n_bins: int = 10,
    min_bin_count: int = 2,
) -> dict[str, Any]:
    """Persistable reliability-bin diagnostic for the active model."""
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reliability atlas features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Reliability atlas features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Reliability atlas feature and label counts do not match.")
    if x.shape[0] == 0:
        raise ValueError("Reliability atlas needs at least one sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Reliability atlas features must be finite numbers.")
    n_bins = max(2, int(n_bins))
    min_bin_count = max(1, int(min_bin_count))

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    if not np.all(np.isfinite(prepared)):
        raise ValueError("Reliability atlas preprocessed features must be finite.")
    probabilities = predict_probability(model, prepared).reshape(-1).astype(np.float64)
    if probabilities.shape[0] != x.shape[0]:
        raise ValueError("Model returned a different number of probabilities than input rows.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model probabilities must be finite.")
    if np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7)):
        raise ValueError("Model probabilities must be between 0 and 1.")
    clipped_probability_count = int(np.count_nonzero((probabilities < 0.0) | (probabilities > 1.0)))
    probabilities = np.clip(probabilities, 0.0, 1.0)
    diagnostics = probability_diagnostics(y, probabilities, n_bins=n_bins)
    bins = [_bin_row(item, min_bin_count=min_bin_count) for item in diagnostics.get("calibration_bins", [])]
    ranked_bins = sorted(bins, key=lambda item: (-float(item["absolute_error"]), -float(item["weighted_error"]), -int(item["count"])))
    impact_bins = sorted(bins, key=lambda item: (-float(item["weighted_error"]), -float(item["absolute_error"]), -int(item["count"])))
    underconfident_bins = [item for item in ranked_bins if item["calibration_direction"] == "underconfident"]
    overconfident_bins = [item for item in ranked_bins if item["calibration_direction"] == "overconfident"]
    sparse_bins = [item for item in bins if int(item["count"]) < min_bin_count]
    worst_bin = ranked_bins[0] if ranked_bins else None
    ece = float(diagnostics.get("expected_calibration_error", 0.0))
    max_error = float(diagnostics.get("max_calibration_error", 0.0))
    brier = float(diagnostics.get("brier_score", 0.0))
    log_loss = float(diagnostics.get("log_loss", 0.0))
    recommendations = _recommendations(
        ece=ece,
        max_error=max_error,
        sparse_count=len(sparse_bins),
        worst_bin=worst_bin,
        under_count=len(underconfident_bins),
        over_count=len(overconfident_bins),
    )
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dataset_fingerprint": reliability_dataset_fingerprint(x, y),
        "n_bins": n_bins,
        "min_bin_count": min_bin_count,
        "summary": {
            "brier_score": brier,
            "log_loss": log_loss,
            "expected_calibration_error": ece,
            "max_calibration_error": max_error,
            "mean_probability": float(diagnostics.get("mean_probability", 0.0)),
            "label_prevalence": float(diagnostics.get("label_prevalence", 0.0)),
            "predicted_positive_rate": float(diagnostics.get("predicted_positive_rate", 0.0)),
            "clipped_probability_count": clipped_probability_count,
            "bin_count": len(bins),
            "sparse_bin_count": len(sparse_bins),
            "overconfident_bin_count": len(overconfident_bins),
            "underconfident_bin_count": len(underconfident_bins),
            "worst_bin": _compact_bin(worst_bin),
            "risk_level": _risk_level(ece, max_error, sparse_bins),
            "recommendation": recommendations[0]["action"] if recommendations else None,
        },
        "bins": bins,
        "worst_bins": ranked_bins[:6],
        "highest_impact_bins": impact_bins[:6],
        "overconfident_bins": overconfident_bins[:6],
        "underconfident_bins": underconfident_bins[:6],
        "sparse_bins": sparse_bins[:6],
        "quantiles_by_class": diagnostics.get("quantiles_by_class", {}),
        "recommendations": recommendations,
    }


def format_reliability_atlas_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    return (
        "Reliability atlas: "
        f"risk={summary.get('risk_level', '-')}, "
        f"ECE={float(summary.get('expected_calibration_error', 0.0)):.4f}, "
        f"max_err={float(summary.get('max_calibration_error', 0.0)):.4f}, "
        f"Brier={float(summary.get('brier_score', 0.0)):.4f}, "
        f"bins={int(summary.get('bin_count', 0))}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def reliability_dataset_fingerprint(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> str:
    """Return a stable fingerprint for the raw labeled rows behind a diagnostic."""
    try:
        x = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reliability atlas features must be finite numbers.") from exc
    y = _validate_labels(labels)
    if x.ndim != 2:
        raise ValueError("Reliability atlas features must be a 2D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Reliability atlas feature and label counts do not match.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Reliability atlas features must be finite numbers.")
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(value) for value in x.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    hasher.update(str(tuple(int(value) for value in y.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(y, dtype=np.int8).tobytes())
    return hasher.hexdigest()


def _validate_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reliability atlas labels must be binary 0/1.") from exc
    if not np.all(np.isfinite(y)):
        raise ValueError("Reliability atlas labels must be binary 0/1.")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Reliability atlas labels must be binary 0/1.")
    return y.astype(np.int32)


def _bin_row(item: dict[str, Any], *, min_bin_count: int) -> dict[str, Any]:
    accuracy = float(item.get("accuracy", 0.0))
    confidence = float(item.get("confidence", 0.0))
    absolute_error = float(item.get("absolute_error", abs(accuracy - confidence)))
    signed_error = accuracy - confidence
    if signed_error > 0.02:
        direction = "underconfident"
    elif signed_error < -0.02:
        direction = "overconfident"
    else:
        direction = "aligned"
    count = int(item.get("count", 0))
    return {
        "left": float(item.get("left", 0.0)),
        "right": float(item.get("right", 1.0)),
        "count": count,
        "weight": float(item.get("weight", 0.0)),
        "accuracy": accuracy,
        "confidence": confidence,
        "signed_error": float(signed_error),
        "absolute_error": absolute_error,
        "weighted_error": float(item.get("weight", 0.0)) * absolute_error,
        "calibration_direction": direction,
        "is_sparse": count < min_bin_count,
    }


def _compact_bin(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "left": item["left"],
        "right": item["right"],
        "count": item["count"],
        "accuracy": item["accuracy"],
        "confidence": item["confidence"],
        "absolute_error": item["absolute_error"],
        "calibration_direction": item["calibration_direction"],
    }


def _risk_level(ece: float, max_error: float, sparse_bins: list[dict[str, Any]]) -> str:
    if ece >= 0.12 or max_error >= 0.35:
        return "high"
    if ece >= 0.08 or max_error >= 0.25 or sparse_bins:
        return "medium"
    return "low"


def _recommendations(
    *,
    ece: float,
    max_error: float,
    sparse_count: int,
    worst_bin: dict[str, Any] | None,
    under_count: int,
    over_count: int,
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

    if ece >= 0.08 or max_error >= 0.25:
        add(
            90.0,
            "high" if ece >= 0.12 or max_error >= 0.35 else "medium",
            "calibration",
            "Run calibration repair before probability use",
            f"ECE={ece:.3f}, max bin error={max_error:.3f}.",
            "Run Calibration repair and compare raw, Platt, and isotonic probability quality.",
        )
    if worst_bin is not None and float(worst_bin.get("absolute_error", 0.0)) >= 0.20:
        add(
            72.0,
            "medium",
            "bin_review",
            "Inspect the worst reliability bin",
            f"Bin [{worst_bin['left']:.2f}, {worst_bin['right']:.2f}) has |accuracy-confidence|={worst_bin['absolute_error']:.3f}.",
            "Inspect rows in this probability range or adjust how probabilities are communicated.",
        )
    if over_count > under_count:
        add(
            58.0,
            "medium",
            "communication",
            "Warn about overconfident probabilities",
            "More reliability bins are overconfident than underconfident.",
            "Avoid presenting probabilities as calibrated until repair or fresh validation confirms them.",
        )
    elif under_count > over_count:
        add(
            50.0,
            "low",
            "communication",
            "Check underconfident probability ranges",
            "More reliability bins are underconfident than overconfident.",
            "Consider whether thresholding is fine but probability communication is too conservative.",
        )
    if sparse_count:
        add(
            46.0,
            "low",
            "evidence",
            "Treat sparse reliability bins cautiously",
            f"{sparse_count} bin(s) have fewer rows than the configured minimum.",
            "Use more validation rows before relying on fine-grained bin conclusions.",
        )
    if not recs:
        add(
            30.0,
            "low",
            "promotion",
            "Keep reliability evidence with the model",
            "Calibration bins show no major local reliability warning.",
            "Export the report or model sidecar so the calibration evidence is retained.",
        )
    recs = sorted(recs, key=lambda item: (-item["priority_score"], item["category"], item["title"]))
    for rank, item in enumerate(recs, start=1):
        item["rank"] = rank
    return recs[:6]
