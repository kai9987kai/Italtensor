from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def run_fairness_diagnostics(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    min_group_size: int = 2,
) -> dict[str, Any]:
    """Audit subgroup metric gaps across automatically derived feature groups."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("Fairness diagnostics features must be a 2D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Fairness diagnostics features must be finite numbers.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Fairness diagnostics feature and label counts do not match.")
    if x.shape[0] < 2:
        raise ValueError("Fairness diagnostics need at least two samples.")
    if set(int(item) for item in np.unique(y)) - {0, 1}:
        raise ValueError("Fairness diagnostics require binary labels 0 or 1.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Fairness diagnostics threshold must be between 0 and 1.")
    if int(min_group_size) < 1:
        raise ValueError("min_group_size must be at least 1.")

    prepared = preprocessor.transform(x) if preprocessor is not None else x
    probabilities = predict_probability(model, prepared)
    if probabilities.shape[0] != y.shape[0]:
        raise ValueError("Model returned a probability count that does not match the dataset.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    predictions = (probabilities >= float(threshold)).astype(np.int32)

    feature_reports = []
    for feature_index in range(x.shape[1]):
        groups = _feature_groups(x[:, feature_index], feature_index, int(min_group_size))
        if len(groups) < 2:
            continue
        group_reports = [
            _group_metrics(group, y, predictions, probabilities)
            for group in groups
        ]
        feature_reports.append(_feature_report(feature_index, group_reports))

    ranked_features = sorted(
        feature_reports,
        key=lambda item: (
            -float(item["max_disparity"]),
            -float(item.get("equalized_odds_gap") or 0.0),
            int(item["feature_index"]),
        ),
    )
    worst = ranked_features[0] if ranked_features else None
    return {
        "sample_count": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "threshold": float(threshold),
        "min_group_size": int(min_group_size),
        "overall": _overall_metrics(y, predictions, probabilities),
        "features": ranked_features,
        "summary": {
            "evaluated_feature_count": int(len(ranked_features)),
            "worst_feature": int(worst["feature_index"]) if worst else None,
            "worst_metric": worst.get("worst_metric") if worst else None,
            "max_disparity": float(worst["max_disparity"]) if worst else 0.0,
            "max_selection_rate_gap": _max_gap(ranked_features, "selection_rate_gap"),
            "max_equal_opportunity_gap": _max_gap(ranked_features, "equal_opportunity_gap"),
            "max_equalized_odds_gap": _max_gap(ranked_features, "equalized_odds_gap"),
            "max_accuracy_gap": _max_gap(ranked_features, "accuracy_gap"),
            "warning": _warning(y, ranked_features),
        },
    }


def format_fairness_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    worst_feature = summary.get("worst_feature")
    feature_text = "-" if worst_feature is None else f"x{int(worst_feature) + 1}"
    return (
        "Fairness audit: "
        f"features={int(summary.get('evaluated_feature_count', 0))}, "
        f"worst={feature_text}, "
        f"metric={summary.get('worst_metric') or '-'}, "
        f"gap={float(summary.get('max_disparity', 0.0)):.4f}, "
        f"EOdds={float(summary.get('max_equalized_odds_gap', 0.0)):.4f}, "
        f"DP={float(summary.get('max_selection_rate_gap', 0.0)):.4f}"
    )


def _feature_groups(values: np.ndarray, feature_index: int, min_group_size: int) -> list[dict[str, Any]]:
    unique = np.unique(values)
    groups: list[dict[str, Any]] = []
    if 2 <= unique.shape[0] <= 4:
        for value in unique:
            mask = values == value
            if int(np.sum(mask)) >= min_group_size:
                groups.append(
                    {
                        "group_id": f"x{feature_index + 1}={float(value):.4g}",
                        "label": f"x{feature_index + 1}={float(value):.4g}",
                        "feature_index": int(feature_index),
                        "kind": "exact",
                        "value": float(value),
                        "mask": mask,
                    }
                )
        return groups

    median = float(np.median(values))
    low = values <= median
    high = values > median
    if int(np.sum(low)) >= min_group_size:
        groups.append(
            {
                "group_id": f"x{feature_index + 1}<=median",
                "label": f"x{feature_index + 1}<={median:.4g}",
                "feature_index": int(feature_index),
                "kind": "lower_half",
                "right": median,
                "mask": low,
            }
        )
    if int(np.sum(high)) >= min_group_size:
        groups.append(
            {
                "group_id": f"x{feature_index + 1}>median",
                "label": f"x{feature_index + 1}>{median:.4g}",
                "feature_index": int(feature_index),
                "kind": "upper_half",
                "left": median,
                "mask": high,
            }
        )
    return groups


def _group_metrics(
    group: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(group["mask"], dtype=bool)
    y = labels[mask]
    pred = predictions[mask]
    probs = probabilities[mask]
    count = int(y.shape[0])
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    result = {key: value for key, value in group.items() if key != "mask"}
    result.update(
        {
            "count": count,
            "weight": float(count / labels.shape[0]),
            "label_prevalence": float(np.mean(y == 1)) if count else 0.0,
            "selection_rate": float(np.mean(pred == 1)) if count else 0.0,
            "mean_probability": float(np.mean(probs)) if count else 0.0,
            "accuracy": float(np.mean(pred == y)) if count else 0.0,
            "precision": _safe_rate(tp, tp + fp),
            "recall": _safe_rate(tp, tp + fn),
            "true_positive_rate": _safe_rate(tp, tp + fn),
            "false_negative_rate": _safe_rate(fn, tp + fn),
            "true_negative_rate": _safe_rate(tn, tn + fp),
            "false_positive_rate": _safe_rate(fp, tn + fp),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        }
    )
    return result


def _overall_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return _group_metrics(
        {
            "group_id": "overall",
            "label": "overall",
            "feature_index": -1,
            "kind": "overall",
            "mask": np.ones(labels.shape[0], dtype=bool),
        },
        labels,
        predictions,
        probabilities,
    )


def _feature_report(feature_index: int, groups: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = {
        "selection_rate_gap": _gap(groups, "selection_rate"),
        "equal_opportunity_gap": _gap(groups, "true_positive_rate"),
        "false_positive_rate_gap": _gap(groups, "false_positive_rate"),
        "false_negative_rate_gap": _gap(groups, "false_negative_rate"),
        "accuracy_gap": _gap(groups, "accuracy"),
        "prevalence_gap": _gap(groups, "label_prevalence"),
    }
    equalized_odds_values = [
        value
        for value in (gaps["equal_opportunity_gap"], gaps["false_positive_rate_gap"])
        if value is not None
    ]
    gaps["equalized_odds_gap"] = max(equalized_odds_values) if equalized_odds_values else None
    ranked_gaps = [
        (key, float(value))
        for key, value in gaps.items()
        if value is not None
    ]
    worst_metric, max_disparity = max(ranked_gaps, key=lambda item: item[1]) if ranked_gaps else ("none", 0.0)
    return {
        "feature_index": int(feature_index),
        "group_count": int(len(groups)),
        "groups": groups,
        **gaps,
        "worst_metric": worst_metric,
        "max_disparity": float(max_disparity),
        "risk_flags": _risk_flags(gaps, float(max_disparity)),
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _gap(groups: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(group[metric]) for group in groups if group.get(metric) is not None]
    if len(values) < 2:
        return None
    return float(max(values) - min(values))


def _max_gap(features: list[dict[str, Any]], metric: str) -> float:
    values = [float(item[metric]) for item in features if item.get(metric) is not None]
    return float(max(values)) if values else 0.0


def _risk_flags(gaps: dict[str, float | None], max_disparity: float) -> list[str]:
    flags: list[str] = []
    if (gaps.get("selection_rate_gap") or 0.0) >= 0.25:
        flags.append("selection_rate_gap")
    if (gaps.get("equalized_odds_gap") or 0.0) >= 0.25:
        flags.append("equalized_odds_gap")
    if (gaps.get("accuracy_gap") or 0.0) >= 0.2:
        flags.append("accuracy_gap")
    if max_disparity >= 0.4:
        flags.append("large_gap")
    return flags


def _warning(labels: np.ndarray, features: list[dict[str, Any]]) -> str | None:
    messages: list[str] = []
    unique = set(int(item) for item in np.unique(labels))
    if unique == {0}:
        messages.append("All labels are negative; true-positive and false-negative gaps are unavailable.")
    elif unique == {1}:
        messages.append("All labels are positive; false-positive gaps are unavailable.")
    if not features:
        messages.append("No feature produced at least two usable groups.")
    messages.append("This is a statistical disparity screen, not a legal fairness determination.")
    return " ".join(messages)
