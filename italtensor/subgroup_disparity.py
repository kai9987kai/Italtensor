from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


DISPARITY_METRICS = (
    "false_negative_rate",
    "false_positive_rate",
    "recall",
    "predicted_positive_rate",
    "accuracy",
    "mean_probability",
)


def run_subgroup_disparity_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    min_group_size: int = 2,
    max_bins: int = 4,
) -> dict[str, Any]:
    """Compare automatically derived numeric subgroups against their complements."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Subgroup disparity features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Subgroup disparity features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Subgroup disparity feature and label counts do not match.")
    if x.shape[0] < 2:
        raise ValueError("Subgroup disparity diagnostics need at least two samples.")
    if set(int(item) for item in np.unique(y)) - {0, 1}:
        raise ValueError("Subgroup disparity diagnostics require binary labels 0 or 1.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Subgroup disparity threshold must be between 0 and 1.")
    if int(min_group_size) < 1:
        raise ValueError("min_group_size must be at least 1.")
    if int(max_bins) < 2:
        raise ValueError("max_bins must be at least 2.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    if probabilities.shape[0] != y.shape[0]:
        raise ValueError("Model returned a probability count that does not match the dataset.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    predictions = (probabilities >= float(threshold)).astype(np.int32)

    feature_reports: list[dict[str, Any]] = []
    subgroups: list[dict[str, Any]] = []
    for feature_index in range(x.shape[1]):
        slices = _candidate_slices(x[:, feature_index], feature_index, int(max_bins))
        reports = [
            _slice_report(candidate, y, predictions, probabilities)
            for candidate in slices
            if _usable_candidate(candidate["mask"], x.shape[0], int(min_group_size))
        ]
        if not reports:
            continue
        reports = sorted(reports, key=lambda item: (-float(item["risk_score"]), int(item["count"]), item["label"]))
        worst = reports[0]
        feature_reports.append(
            {
                "feature_index": int(feature_index),
                "slice_count": int(len(reports)),
                "max_disparity": float(worst["risk_score"]),
                "worst_metric": worst["worst_metric"],
                "worst_subgroup": worst["label"],
                "subgroups": reports[:8],
            }
        )
        subgroups.extend(reports)

    feature_reports = sorted(
        feature_reports,
        key=lambda item: (-float(item["max_disparity"]), int(item["feature_index"])),
    )
    ranked_subgroups = sorted(
        subgroups,
        key=lambda item: (-float(item["risk_score"]), int(item["feature_index"]), item["label"]),
    )
    worst = ranked_subgroups[0] if ranked_subgroups else None
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "min_group_size": int(min_group_size),
        "overall": _metrics(y, predictions, probabilities),
        "features": feature_reports,
        "subgroups": ranked_subgroups[:20],
        "summary": {
            "evaluated_feature_count": int(len(feature_reports)),
            "evaluated_subgroup_count": int(len(ranked_subgroups)),
            "worst_feature": int(worst["feature_index"]) if worst else None,
            "worst_subgroup": worst["label"] if worst else None,
            "worst_metric": worst["worst_metric"] if worst else None,
            "max_disparity": float(worst["risk_score"]) if worst else 0.0,
            "max_false_negative_rate_gap": _max_gap(ranked_subgroups, "false_negative_rate_gap"),
            "max_false_positive_rate_gap": _max_gap(ranked_subgroups, "false_positive_rate_gap"),
            "max_recall_gap": _max_gap(ranked_subgroups, "recall_gap"),
            "max_predicted_positive_rate_gap": _max_gap(ranked_subgroups, "predicted_positive_rate_gap"),
            "max_accuracy_gap": _max_gap(ranked_subgroups, "accuracy_gap"),
            "warning": _warning(y, ranked_subgroups),
        },
    }


def format_subgroup_disparity_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    worst_feature = summary.get("worst_feature")
    feature_text = "-" if worst_feature is None else f"x{int(worst_feature) + 1}"
    return (
        "Subgroup disparity: "
        f"features={int(summary.get('evaluated_feature_count', 0))}, "
        f"groups={int(summary.get('evaluated_subgroup_count', 0))}, "
        f"worst={feature_text}, "
        f"metric={summary.get('worst_metric') or '-'}, "
        f"gap={float(summary.get('max_disparity', 0.0)):.4f}"
    )


def _candidate_slices(values: np.ndarray, feature_index: int, max_bins: int) -> list[dict[str, Any]]:
    unique = np.unique(values)
    if 2 <= unique.shape[0] <= max_bins:
        return [
            {
                "feature_index": int(feature_index),
                "label": f"x{feature_index + 1}={float(value):.4g}",
                "kind": "exact",
                "value": float(value),
                "mask": values == value,
            }
            for value in unique
        ]

    quantiles = np.linspace(0.0, 1.0, max_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.shape[0] < 3:
        return []
    slices: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == edges.shape[0] - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        slices.append(
            {
                "feature_index": int(feature_index),
                "label": f"x{feature_index + 1}[{float(left):.4g}, {float(right):.4g}]",
                "kind": "quantile_bin",
                "left": float(left),
                "right": float(right),
                "mask": mask,
            }
        )
    return slices


def _usable_candidate(mask: np.ndarray, total: int, min_group_size: int) -> bool:
    count = int(np.sum(mask))
    complement_count = int(total - count)
    return count >= min_group_size and complement_count >= min_group_size


def _slice_report(
    candidate: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(candidate["mask"], dtype=bool)
    complement = ~mask
    subgroup = _metrics(labels[mask], predictions[mask], probabilities[mask])
    complement_metrics = _metrics(labels[complement], predictions[complement], probabilities[complement])
    disparities = _disparities(subgroup, complement_metrics)
    ranked = [(key, value) for key, value in disparities.items() if value is not None]
    worst_metric, risk_score = max(ranked, key=lambda item: float(item[1])) if ranked else ("none", 0.0)
    report = {key: value for key, value in candidate.items() if key != "mask"}
    report.update(
        {
            "count": int(np.sum(mask)),
            "coverage": float(np.mean(mask)),
            "subgroup": subgroup,
            "complement": complement_metrics,
            **disparities,
            "worst_metric": worst_metric,
            "risk_score": float(risk_score),
            "risk_flags": _risk_flags(disparities, float(risk_score)),
            "warnings": _slice_warnings(labels[mask], labels[complement]),
        }
    )
    return report


def _metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    count = int(labels.shape[0])
    if count == 0:
        return {
            "count": 0,
            "label_prevalence": None,
            "predicted_positive_rate": None,
            "mean_probability": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positive_rate": None,
            "false_negative_rate": None,
            "true_negative_rate": None,
            "false_positive_rate": None,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    return {
        "count": count,
        "label_prevalence": float(np.mean(labels == 1)),
        "predicted_positive_rate": float(np.mean(predictions == 1)),
        "mean_probability": float(np.mean(probabilities)),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positive_rate": recall,
        "false_negative_rate": _safe_rate(fn, tp + fn),
        "true_negative_rate": _safe_rate(tn, tn + fp),
        "false_positive_rate": _safe_rate(fp, tn + fp),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _disparities(subgroup: dict[str, Any], complement: dict[str, Any]) -> dict[str, float | None]:
    return {
        f"{metric}_gap": _absolute_gap(subgroup.get(metric), complement.get(metric))
        for metric in (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "false_negative_rate",
            "false_positive_rate",
            "predicted_positive_rate",
            "mean_probability",
            "label_prevalence",
        )
    }


def _absolute_gap(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(abs(float(left) - float(right)))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    return float(2.0 * precision * recall / denominator)


def _max_gap(subgroups: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in subgroups if item.get(key) is not None]
    return float(max(values)) if values else 0.0


def _risk_flags(disparities: dict[str, float | None], risk_score: float) -> list[str]:
    flags: list[str] = []
    if (disparities.get("false_negative_rate_gap") or 0.0) >= 0.25:
        flags.append("fnr_gap")
    if (disparities.get("false_positive_rate_gap") or 0.0) >= 0.25:
        flags.append("fpr_gap")
    if (disparities.get("predicted_positive_rate_gap") or 0.0) >= 0.25:
        flags.append("selection_gap")
    if (disparities.get("accuracy_gap") or 0.0) >= 0.2:
        flags.append("accuracy_gap")
    if risk_score >= 0.4:
        flags.append("large_gap")
    return flags


def _slice_warnings(subgroup_labels: np.ndarray, complement_labels: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if len(set(int(item) for item in np.unique(subgroup_labels))) < 2:
        warnings.append("one_class_subgroup")
    if len(set(int(item) for item in np.unique(complement_labels))) < 2:
        warnings.append("one_class_complement")
    if subgroup_labels.shape[0] < 10:
        warnings.append("small_subgroup")
    return warnings


def _warning(labels: np.ndarray, subgroups: list[dict[str, Any]]) -> str:
    messages: list[str] = []
    unique = set(int(item) for item in np.unique(labels))
    if unique == {0}:
        messages.append("All labels are negative; true-positive and false-negative disparities are unavailable.")
    elif unique == {1}:
        messages.append("All labels are positive; false-positive disparities are unavailable.")
    if not subgroups:
        messages.append("No feature produced a usable subgroup/complement split.")
    messages.append("Numeric feature slices are proxy subgroup diagnostics, not proof of protected-class fairness compliance.")
    return " ".join(messages)
