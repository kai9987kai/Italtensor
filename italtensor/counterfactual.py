from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


@dataclass(frozen=True)
class CounterfactualResult:
    success: bool
    original_probability: float
    original_label: int
    target_label: int
    candidate: list[float] | None
    candidate_probability: float | None
    delta: list[float] | None
    normalized_l1: float | None
    changed_features: list[dict[str, float | int]]
    evaluated: int
    reason: str


def find_counterfactual(
    model: Any,
    vector: Sequence[float],
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    target_label: int | None = None,
    max_steps: int = 18,
    samples_per_step: int = 48,
    seed: int = 42,
) -> CounterfactualResult:
    """Search for a nearby numeric vector that flips the model decision."""
    raw = np.asarray(vector, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        raise ValueError("Counterfactual vector cannot be empty.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Counterfactual vector must contain finite numbers.")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    original_probability = _predict_raw(model, raw, preprocessor)
    original_label = int(original_probability >= threshold)
    target = 1 - original_label if target_label is None else int(target_label)
    if target not in (0, 1):
        raise ValueError("target_label must be 0 or 1.")
    if target == original_label:
        return CounterfactualResult(
            success=True,
            original_probability=original_probability,
            original_label=original_label,
            target_label=target,
            candidate=raw.astype(float).tolist(),
            candidate_probability=original_probability,
            delta=[0.0 for _ in range(raw.size)],
            normalized_l1=0.0,
            changed_features=[],
            evaluated=1,
            reason="already_target_label",
        )

    rng = np.random.default_rng(seed)
    scale = _search_scale(raw, preprocessor)
    mutable_indices = _mutable_indices(raw.size, preprocessor)
    best: tuple[float, np.ndarray, float] | None = None
    evaluated = 1
    radii = np.geomspace(0.03, 4.0, num=max(1, int(max_steps)))

    for radius in radii:
        for candidate in _candidate_vectors(raw, scale, mutable_indices, float(radius), samples_per_step, rng):
            probability = _predict_raw(model, candidate, preprocessor)
            evaluated += 1
            if _meets_target(probability, threshold, target):
                distance = _normalized_l1(raw, candidate, scale)
                confidence_gap = abs(float(probability) - float(threshold))
                objective = distance - 0.01 * confidence_gap
                if best is None or objective < best[0]:
                    best = (objective, candidate.copy(), float(probability))

    if best is None:
        return CounterfactualResult(
            success=False,
            original_probability=original_probability,
            original_label=original_label,
            target_label=target,
            candidate=None,
            candidate_probability=None,
            delta=None,
            normalized_l1=None,
            changed_features=[],
            evaluated=evaluated,
            reason="no_flip_found",
        )

    _, candidate, candidate_probability = best
    delta = candidate - raw
    return CounterfactualResult(
        success=True,
        original_probability=original_probability,
        original_label=original_label,
        target_label=target,
        candidate=candidate.astype(float).tolist(),
        candidate_probability=candidate_probability,
        delta=delta.astype(float).tolist(),
        normalized_l1=_normalized_l1(raw, candidate, scale),
        changed_features=_changed_features(raw, candidate, delta),
        evaluated=evaluated,
        reason="flip_found",
    )


def format_counterfactual_result(result: CounterfactualResult, *, max_changes: int = 6) -> str:
    if not result.success:
        return (
            "Counterfactual: no nearby flip found "
            f"(current label={result.original_label}, p={result.original_probability:.4f}, "
            f"target={result.target_label}, evaluated={result.evaluated})."
        )
    if result.reason == "already_target_label":
        return (
            "Counterfactual: no change needed "
            f"(label={result.original_label}, p={result.original_probability:.4f})."
        )
    changes = result.changed_features[:max_changes]
    change_text = "; ".join(
        f"x{int(item['feature_index']) + 1}: {float(item['original']):.4g} -> "
        f"{float(item['counterfactual']):.4g} ({float(item['delta']):+.4g})"
        for item in changes
    )
    if len(result.changed_features) > max_changes:
        change_text += f"; +{len(result.changed_features) - max_changes} more"
    return (
        f"Counterfactual target={result.target_label}: "
        f"p {result.original_probability:.4f} -> {float(result.candidate_probability):.4f}, "
        f"distance={float(result.normalized_l1):.4f}, changes: {change_text}"
    )


def _predict_raw(model: Any, raw: np.ndarray, preprocessor: FeatureStandardizer | None) -> float:
    prepared = preprocessor.transform(raw) if preprocessor is not None else raw.reshape(1, -1)
    return float(predict_probability(model, prepared)[0])


def _search_scale(raw: np.ndarray, preprocessor: FeatureStandardizer | None) -> np.ndarray:
    scale = np.maximum(np.abs(raw), 1.0).astype(np.float32)
    if preprocessor is None:
        return scale
    if preprocessor.selected_indices is None and preprocessor.scale.shape[0] == raw.size:
        return np.maximum(preprocessor.scale.astype(np.float32), 1e-3)
    if preprocessor.selected_indices is not None:
        selected_scale = np.maximum(preprocessor.scale.astype(np.float32), 1e-3)
        for position, raw_index in enumerate(preprocessor.selected_indices):
            if 0 <= raw_index < raw.size and position < selected_scale.shape[0]:
                scale[raw_index] = selected_scale[position]
    return scale


def _mutable_indices(input_dim: int, preprocessor: FeatureStandardizer | None) -> list[int]:
    if preprocessor is not None and preprocessor.selected_indices is not None:
        return [index for index in preprocessor.selected_indices if 0 <= index < input_dim]
    return list(range(input_dim))


def _candidate_vectors(
    raw: np.ndarray,
    scale: np.ndarray,
    mutable_indices: list[int],
    radius: float,
    samples_per_step: int,
    rng: np.random.Generator,
):
    for index in mutable_indices:
        for direction in (-1.0, 1.0):
            candidate = raw.copy()
            candidate[index] += direction * radius * scale[index]
            yield candidate

    all_positive = raw.copy()
    all_negative = raw.copy()
    all_positive[mutable_indices] += radius * scale[mutable_indices]
    all_negative[mutable_indices] -= radius * scale[mutable_indices]
    yield all_positive
    yield all_negative

    random_count = max(0, int(samples_per_step))
    for _ in range(random_count):
        noise = rng.normal(0.0, 1.0, size=raw.size).astype(np.float32)
        mask = np.zeros(raw.size, dtype=np.float32)
        mask[mutable_indices] = 1.0
        candidate = raw + noise * mask * scale * radius
        yield candidate.astype(np.float32)


def _meets_target(probability: float, threshold: float, target_label: int) -> bool:
    if target_label == 1:
        return probability >= threshold
    return probability < threshold


def _normalized_l1(original: np.ndarray, candidate: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sum(np.abs(candidate - original) / np.maximum(scale, 1e-6)))


def _changed_features(original: np.ndarray, candidate: np.ndarray, delta: np.ndarray) -> list[dict[str, float | int]]:
    changed: list[dict[str, float | int]] = []
    for index, value in enumerate(delta):
        if abs(float(value)) <= 1e-6:
            continue
        changed.append(
            {
                "feature_index": index,
                "original": float(original[index]),
                "counterfactual": float(candidate[index]),
                "delta": float(value),
            }
        )
    changed.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return changed
