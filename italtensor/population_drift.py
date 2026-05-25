from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_BINS = 10


def run_population_drift_diagnostics(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    reference_fraction: float = 0.5,
    n_bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Compare early/reference rows with later/current rows for population drift."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Population drift features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Population drift features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Population drift feature and label counts do not match.")
    if x.shape[0] < 6:
        raise ValueError("Population drift diagnostics need at least six rows.")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Population drift diagnostics require binary labels 0 or 1.")
    if not 0.1 <= float(reference_fraction) <= 0.9:
        raise ValueError("Population drift reference_fraction must be between 0.1 and 0.9.")
    if int(n_bins) < 2:
        raise ValueError("Population drift diagnostics need at least two bins.")

    reference_count = int(round(x.shape[0] * float(reference_fraction)))
    reference_count = min(max(reference_count, 2), x.shape[0] - 2)
    reference = x[:reference_count]
    current = x[reference_count:]
    reference_labels = y[:reference_count]
    current_labels = y[reference_count:]

    rows = [
        _feature_drift_row(index, reference[:, index], current[:, index], int(n_bins))
        for index in range(x.shape[1])
    ]
    rows.sort(key=lambda item: (-float(item["risk_score"]), int(item["feature_index"])))
    label_shift = _label_shift(reference_labels, current_labels)
    drifted = [row for row in rows if row["risk_flags"]]
    summary = {
        "top_feature": int(rows[0]["feature_index"]) if rows else None,
        "max_psi": float(max((row["psi"] for row in rows), default=0.0)),
        "max_ks_statistic": float(max((row["ks_statistic"] for row in rows), default=0.0)),
        "max_mean_shift_std": float(max((row["mean_shift_std"] for row in rows), default=0.0)),
        "max_outside_reference_rate": float(max((row["outside_reference_rate"] for row in rows), default=0.0)),
        "drifted_feature_count": int(len(drifted)),
        "label_prevalence_shift": float(label_shift["prevalence_shift"]),
        "warning": _warning(reference.shape[0], current.shape[0], drifted, label_shift),
    }
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "reference_fraction": float(reference_fraction),
        "reference_count": int(reference.shape[0]),
        "current_count": int(current.shape[0]),
        "n_bins": int(n_bins),
        "split_source": "row_order_first_reference_then_current",
        "label_shift": label_shift,
        "features": rows,
        "summary": summary,
    }


def format_population_drift_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    top_feature = summary.get("top_feature")
    top_text = "-" if top_feature is None else f"x{int(top_feature) + 1}"
    return (
        "Population drift: "
        f"top={top_text}, "
        f"max_PSI={float(summary.get('max_psi', 0.0)):.4f}, "
        f"max_KS={float(summary.get('max_ks_statistic', 0.0)):.4f}, "
        f"max_mean_shift={float(summary.get('max_mean_shift_std', 0.0)):.4f}, "
        f"label_shift={float(summary.get('label_prevalence_shift', 0.0)):.4f}, "
        f"drifted={int(summary.get('drifted_feature_count', 0))}"
    )


def _feature_drift_row(index: int, reference: np.ndarray, current: np.ndarray, n_bins: int) -> dict[str, Any]:
    reference_std = float(np.std(reference))
    current_std = float(np.std(current))
    safe_std = reference_std if reference_std >= 1e-6 else 1.0
    mean_shift = abs(float(np.mean(current)) - float(np.mean(reference))) / safe_std
    psi = _population_stability_index(reference, current, n_bins)
    ks = _ks_statistic(reference, current)
    outside_rate = _outside_reference_rate(reference, current)
    variance_ratio = current_std / safe_std
    flags = _risk_flags(psi, ks, mean_shift, outside_rate, variance_ratio)
    risk_score = float(psi + ks + min(mean_shift, 5.0) / 5.0 + outside_rate)
    return {
        "feature_index": int(index),
        "reference_mean": float(np.mean(reference)),
        "current_mean": float(np.mean(current)),
        "reference_std": reference_std,
        "current_std": current_std,
        "std_ratio": float(variance_ratio),
        "psi": float(psi),
        "ks_statistic": float(ks),
        "mean_shift_std": float(mean_shift),
        "outside_reference_rate": float(outside_rate),
        "risk_score": risk_score,
        "risk_flags": flags,
    }


def _population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int) -> float:
    boundaries = np.quantile(reference, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
    boundaries = np.unique(boundaries[np.isfinite(boundaries)])
    edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
    if edges.size < 3:
        edges = np.asarray([-np.inf, np.inf], dtype=np.float32)
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    reference_pct = np.maximum(reference_counts / max(reference.shape[0], 1), epsilon)
    current_pct = np.maximum(current_counts / max(current.shape[0], 1), epsilon)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    values = np.unique(np.concatenate((reference, current)))
    if values.size == 0:
        return 0.0
    reference_sorted = np.sort(reference)
    current_sorted = np.sort(current)
    reference_cdf = np.searchsorted(reference_sorted, values, side="right") / reference.shape[0]
    current_cdf = np.searchsorted(current_sorted, values, side="right") / current.shape[0]
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def _outside_reference_rate(reference: np.ndarray, current: np.ndarray) -> float:
    left = float(np.min(reference))
    right = float(np.max(reference))
    outside = (current < left) | (current > right)
    return float(np.mean(outside))


def _label_shift(reference_labels: np.ndarray, current_labels: np.ndarray) -> dict[str, Any]:
    reference_prevalence = float(np.mean(reference_labels == 1))
    current_prevalence = float(np.mean(current_labels == 1))
    return {
        "reference_prevalence": reference_prevalence,
        "current_prevalence": current_prevalence,
        "prevalence_shift": float(abs(current_prevalence - reference_prevalence)),
        "reference_counts": {
            "0": int(np.sum(reference_labels == 0)),
            "1": int(np.sum(reference_labels == 1)),
        },
        "current_counts": {
            "0": int(np.sum(current_labels == 0)),
            "1": int(np.sum(current_labels == 1)),
        },
    }


def _risk_flags(
    psi: float,
    ks: float,
    mean_shift: float,
    outside_rate: float,
    variance_ratio: float,
) -> list[str]:
    flags: list[str] = []
    if psi >= 0.25:
        flags.append("major_psi_shift")
    elif psi >= 0.10:
        flags.append("moderate_psi_shift")
    if ks >= 0.30:
        flags.append("high_ks_distance")
    elif ks >= 0.20:
        flags.append("moderate_ks_distance")
    if mean_shift >= 1.0:
        flags.append("mean_shift")
    if outside_rate >= 0.10:
        flags.append("outside_reference_range")
    if variance_ratio >= 2.0 or variance_ratio <= 0.5:
        flags.append("variance_shift")
    return flags


def _warning(
    reference_count: int,
    current_count: int,
    drifted: list[dict[str, Any]],
    label_shift: dict[str, Any],
) -> str | None:
    warnings: list[str] = []
    if reference_count < 20 or current_count < 20:
        warnings.append("small reference/current split")
    if not drifted and float(label_shift["prevalence_shift"]) < 0.10:
        warnings.append("no large univariate drift detected")
    if float(label_shift["prevalence_shift"]) >= 0.20:
        warnings.append("label prevalence shifted")
    return "; ".join(warnings) if warnings else None
