from __future__ import annotations

from typing import Any

import numpy as np

from .data import Dataset, validate_dataset


def audit_dataset(features: list[list[float]] | np.ndarray, labels: list[int] | np.ndarray) -> dict[str, Any]:
    dataset = _as_dataset(features, labels)
    x = dataset.features
    y = dataset.labels
    class_counts = {
        "0": int(np.sum(y == 0)),
        "1": int(np.sum(y == 1)),
    }
    minority = min(class_counts.values())
    majority = max(class_counts.values())
    duplicate_count, conflict_count = _duplicate_and_conflict_counts(x, y)
    constant_features = _constant_features(x)
    high_correlations = _high_correlations(x)
    warnings = _audit_warnings(
        sample_count=dataset.sample_count,
        class_counts=class_counts,
        duplicate_count=duplicate_count,
        conflict_count=conflict_count,
        constant_features=constant_features,
        high_correlations=high_correlations,
    )
    return {
        "sample_count": dataset.sample_count,
        "input_dim": dataset.input_dim,
        "class_counts": class_counts,
        "minority_class_count": int(minority),
        "majority_class_count": int(majority),
        "imbalance_ratio": float(majority / max(minority, 1)),
        "duplicate_row_count": duplicate_count,
        "label_conflict_count": conflict_count,
        "constant_features": constant_features,
        "high_correlations": high_correlations,
        "warnings": warnings,
    }


def format_audit_summary(audit: dict[str, Any]) -> str:
    warning_text = "; ".join(audit.get("warnings", [])) or "no warnings"
    return (
        f"Dataset audit: samples={audit.get('sample_count')}, input_dim={audit.get('input_dim')}, "
        f"class_counts={audit.get('class_counts')}, imbalance={float(audit.get('imbalance_ratio', 0.0)):.2f}, "
        f"duplicates={audit.get('duplicate_row_count')}, conflicts={audit.get('label_conflict_count')}, "
        f"constant_features={audit.get('constant_features')}, high_corr_pairs={len(audit.get('high_correlations', []))}, "
        f"warnings={warning_text}."
    )


def _as_dataset(features: list[list[float]] | np.ndarray, labels: list[int] | np.ndarray) -> Dataset:
    feature_list = np.asarray(features, dtype=np.float32).tolist()
    label_list = np.asarray(labels, dtype=np.int32).reshape(-1).tolist()
    return validate_dataset(feature_list, label_list, min_samples=1)


def _duplicate_and_conflict_counts(features: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    seen: dict[tuple[float, ...], set[int]] = {}
    duplicate_count = 0
    for row, label in zip(features, labels, strict=True):
        key = tuple(round(float(value), 8) for value in row)
        if key in seen:
            duplicate_count += 1
            seen[key].add(int(label))
        else:
            seen[key] = {int(label)}
    conflict_count = sum(1 for labels_for_row in seen.values() if len(labels_for_row) > 1)
    return duplicate_count, conflict_count


def _constant_features(features: np.ndarray) -> list[int]:
    if features.shape[0] < 2:
        return list(range(features.shape[1]))
    std = features.std(axis=0)
    return [int(index) for index, value in enumerate(std) if float(value) < 1e-8]


def _high_correlations(features: np.ndarray, threshold: float = 0.95, limit: int = 12) -> list[dict[str, float | int]]:
    if features.shape[0] < 3 or features.shape[1] < 2:
        return []
    std = features.std(axis=0)
    variable_indices = [index for index, value in enumerate(std) if float(value) >= 1e-8]
    if len(variable_indices) < 2:
        return []
    variable = features[:, variable_indices]
    corr = np.corrcoef(variable, rowvar=False)
    pairs: list[dict[str, float | int]] = []
    for left in range(corr.shape[0]):
        for right in range(left + 1, corr.shape[1]):
            value = float(corr[left, right])
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "left": int(variable_indices[left]),
                        "right": int(variable_indices[right]),
                        "correlation": value,
                    }
                )
    pairs.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return pairs[:limit]


def _audit_warnings(
    *,
    sample_count: int,
    class_counts: dict[str, int],
    duplicate_count: int,
    conflict_count: int,
    constant_features: list[int],
    high_correlations: list[dict[str, float | int]],
) -> list[str]:
    warnings: list[str] = []
    if sample_count < 20:
        warnings.append("very small dataset")
    minority = min(class_counts.values())
    majority = max(class_counts.values())
    if minority == 0:
        warnings.append("missing one class")
    elif majority / minority >= 4.0:
        warnings.append("strong class imbalance")
    if duplicate_count:
        warnings.append("duplicate feature rows")
    if conflict_count:
        warnings.append("same features appear with both labels")
    if constant_features:
        warnings.append("constant features")
    if high_correlations:
        warnings.append("highly correlated features")
    return warnings
