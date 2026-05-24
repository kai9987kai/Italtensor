from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import Dataset, DataValidationError, dataset_from_jsonable, dataset_to_jsonable, validate_dataset

SCHEMA_VERSION = 1
DEFAULT_TRAINING_DEFAULTS = {"epochs": 50, "batch_size": 16, "trials": 8, "feature_map": "linear"}


@dataclass(frozen=True)
class PresetInfo:
    key: str
    name: str
    description: str
    default_samples: int
    min_samples: int = 4
    input_dim: int = 2
    recommended_feature_map: str = "linear"
    feature_names: tuple[str, ...] = ("x1", "x2")
    label_names: tuple[str, str] = ("negative", "positive")
    training_defaults: dict[str, object] = field(default_factory=lambda: DEFAULT_TRAINING_DEFAULTS.copy())
    prediction_examples: tuple[dict[str, object], ...] = field(default_factory=tuple)


BUILT_IN_PRESETS: tuple[PresetInfo, ...] = (
    PresetInfo(
        key="linear_blobs",
        name="Linear blobs",
        description="Two separable 2D Gaussian clusters for quick sanity checks.",
        default_samples=80,
        recommended_feature_map="linear",
        prediction_examples=(
            {"name": "Likely class 0", "features": [-1.25, -1.0], "expected_label": 0},
            {"name": "Likely class 1", "features": [1.25, 1.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="xor",
        name="XOR pattern",
        description="A nonlinear 2D pattern that rewards hidden layers over a linear boundary.",
        default_samples=96,
        recommended_feature_map="quadratic",
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Low-low", "features": [-1.0, -1.0], "expected_label": 0},
            {"name": "Low-high", "features": [-1.0, 1.0], "expected_label": 1},
            {"name": "High-low", "features": [1.0, -1.0], "expected_label": 1},
            {"name": "High-high", "features": [1.0, 1.0], "expected_label": 0},
        ),
    ),
    PresetInfo(
        key="imbalanced_blobs",
        name="Imbalanced blobs",
        description="A skewed 2D binary dataset for testing class weights and balanced metrics.",
        default_samples=100,
        recommended_feature_map="linear",
        training_defaults={"epochs": 60, "batch_size": 16, "trials": 10, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Common region", "features": [-0.35, -0.1], "expected_label": 0},
            {"name": "Minority region", "features": [1.15, 1.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="signal_plus_noise",
        name="Signal plus noise",
        description="Six features where only the first two drive the label, useful for feature importance.",
        default_samples=120,
        input_dim=6,
        recommended_feature_map="linear",
        feature_names=("signal_a", "signal_b", "noise_1", "noise_2", "noise_3", "noise_4"),
        training_defaults={"epochs": 60, "batch_size": 16, "trials": 10, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Lower score", "features": [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "expected_label": 0},
            {"name": "Higher score", "features": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="concentric_rings",
        name="Concentric rings",
        description="A radial nonlinear dataset for trying RFF feature maps.",
        default_samples=120,
        recommended_feature_map="rff",
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 16, "feature_map": "rff"},
        prediction_examples=(
            {"name": "Inner ring", "features": [0.65, 0.0], "expected_label": 0},
            {"name": "Outer ring", "features": [1.35, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="two_moons",
        name="Two moons",
        description="Interleaving crescent shapes for nonlinear boundary experiments.",
        default_samples=120,
        recommended_feature_map="rff",
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 16, "feature_map": "rff"},
        prediction_examples=(
            {"name": "Upper arc", "features": [0.0, 1.0], "expected_label": 0},
            {"name": "Lower arc", "features": [1.0, -0.55], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="rare_event_signal",
        name="Rare event signal",
        description="A heavily imbalanced dataset with a compact positive region.",
        default_samples=160,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("region_x", "region_y", "background_1", "background_2"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Typical negative", "features": [0.0, 0.0, 0.0, 0.0], "expected_label": 0},
            {"name": "Rare positive", "features": [1.7, 1.5, 0.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="overlapping_margin",
        name="Overlapping margin",
        description="Partly overlapping diagonal clusters for uncertainty and abstention experiments.",
        default_samples=140,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Likely class 0", "features": [-0.85, -0.7], "expected_label": 0},
            {"name": "Ambiguous margin", "features": [0.05, 0.0], "expected_label": None},
            {"name": "Likely class 1", "features": [0.85, 0.7], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="noisy_labels",
        name="Noisy labels",
        description="Mostly separable diagonal blobs with controlled label flips for robustness checks.",
        default_samples=140,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clean negative region", "features": [-1.1, -0.9], "expected_label": 0},
            {"name": "Clean positive region", "features": [1.1, 0.9], "expected_label": 1},
            {"name": "Ambiguous noisy margin", "features": [0.0, 0.0], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="sparse_interaction_signal",
        name="Sparse interaction signal",
        description="Sixteen features where a few sparse terms and one interaction drive the label.",
        default_samples=180,
        input_dim=16,
        recommended_feature_map="quadratic",
        feature_names=tuple(f"feature_{index + 1}" for index in range(16)),
        training_defaults={
            "epochs": 90,
            "batch_size": 16,
            "trials": 16,
            "feature_map": "quadratic",
            "l1_penalty": 0.001,
            "feature_selection_k": 6,
        },
        prediction_examples=(
            {
                "name": "Sparse negative",
                "features": [-1.0, 0.4, 0.0, -0.8, 0.0, 0.0, 0.0, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "expected_label": 0,
            },
            {
                "name": "Interaction positive",
                "features": [1.0, 1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "expected_label": 1,
            },
        ),
    ),
    PresetInfo(
        key="deployment_drift_probe",
        name="Deployment drift probe",
        description="Compact training distribution with examples that make shifted batch rows easy to flag.",
        default_samples=140,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("signal_x", "signal_y", "stable_noise", "shift_probe"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 10, "feature_map": "linear"},
        prediction_examples=(
            {"name": "In-distribution negative", "features": [-0.9, -0.7, 0.0, 0.0], "expected_label": 0},
            {"name": "In-distribution positive", "features": [0.9, 0.7, 0.0, 0.0], "expected_label": 1},
            {"name": "Drift review row", "features": [0.0, 0.0, 4.5, -4.5], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="active_learning_margin",
        name="Active learning margin",
        description="Dense linear classes with an ambiguous boundary belt for query-ranking and counterfactual recourse.",
        default_samples=160,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clear negative", "features": [-1.2, -0.55], "expected_label": 0},
            {"name": "Boundary query", "features": [0.03, -0.02], "expected_label": None},
            {"name": "Clear positive", "features": [1.2, 0.55], "expected_label": 1},
            {"name": "Drifting candidate", "features": [0.0, 3.8], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="spurious_shortcut",
        name="Spurious shortcut",
        description="A strong shortcut feature works during training but invites robustness and drift stress tests.",
        default_samples=180,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("stable_signal", "context_noise", "shortcut_signal"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Shortcut negative", "features": [-0.7, 0.0, -1.2], "expected_label": 0},
            {"name": "Shortcut positive", "features": [0.7, 0.0, 1.2], "expected_label": 1},
            {"name": "Shortcut conflict", "features": [0.8, 0.0, -1.2], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="subgroup_blind_spot",
        name="Subgroup blind spot",
        description="A minority subgroup follows a different rule, useful for slice diagnostics and interaction models.",
        default_samples=180,
        input_dim=3,
        recommended_feature_map="quadratic",
        feature_names=("primary_signal", "subgroup_marker", "context_noise"),
        training_defaults={"epochs": 90, "batch_size": 16, "trials": 16, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Majority negative", "features": [-0.8, 0.0, 0.0], "expected_label": 0},
            {"name": "Majority positive", "features": [0.8, 0.0, 0.0], "expected_label": 1},
            {"name": "Minority flipped rule", "features": [0.8, 1.0, 0.0], "expected_label": 0},
        ),
    ),
    PresetInfo(
        key="cost_sensitive_screening",
        name="Cost-sensitive screening",
        description="Rare positives with overlapping scores for threshold tradeoff and false-negative-cost experiments.",
        default_samples=180,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("risk_score", "secondary_signal", "background_noise"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Likely negative", "features": [-0.6, -0.2, 0.0], "expected_label": 0},
            {"name": "Borderline review", "features": [0.25, 0.1, 0.0], "expected_label": None},
            {"name": "Likely positive", "features": [1.0, 0.7, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="label_audit_traps",
        name="Label audit traps",
        description="Mostly clean separable classes with a small set of flipped labels for sample-review drills.",
        default_samples=160,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clean negative", "features": [-1.3, -1.0], "expected_label": 0},
            {"name": "Clean positive", "features": [1.3, 1.0], "expected_label": 1},
            {"name": "Suspicious positive-shaped negative", "features": [1.2, 0.9], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="proxy_leakage_lab",
        name="Proxy leakage lab",
        description="A label-correlated proxy feature that makes ablation and reliance diagnostics visible.",
        default_samples=180,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("real_signal", "weak_signal", "proxy_code", "background_noise"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Signal negative", "features": [-0.8, -0.2, -1.3, 0.0], "expected_label": 0},
            {"name": "Signal positive", "features": [0.8, 0.2, 1.3, 0.0], "expected_label": 1},
            {"name": "Proxy conflict", "features": [0.8, 0.2, -1.3, 0.0], "expected_label": None},
        ),
    ),
)


def preset_labels() -> list[str]:
    return [preset.name for preset in BUILT_IN_PRESETS]


def preset_by_name(name: str) -> PresetInfo:
    for preset in BUILT_IN_PRESETS:
        if preset.name == name or preset.key == name:
            return preset
    raise ValueError(f"Unknown preset: {name}")


def preset_metadata(name: str) -> dict[str, object]:
    preset = preset_by_name(name)
    return _metadata_from_preset(preset)


def generate_builtin_preset(name: str, *, sample_count: int | None = None, seed: int = 42) -> Dataset:
    preset = preset_by_name(name)
    total = int(sample_count or preset.default_samples)
    if total < preset.min_samples:
        raise ValueError(f"{preset.name} needs at least {preset.min_samples} samples.")

    rng = np.random.default_rng(seed)
    if preset.key == "linear_blobs":
        features, labels = _linear_blobs(total, rng)
    elif preset.key == "xor":
        features, labels = _xor(total, rng)
    elif preset.key == "imbalanced_blobs":
        features, labels = _imbalanced_blobs(total, rng)
    elif preset.key == "signal_plus_noise":
        features, labels = _signal_plus_noise(total, rng)
    elif preset.key == "concentric_rings":
        features, labels = _concentric_rings(total, rng)
    elif preset.key == "two_moons":
        features, labels = _two_moons(total, rng)
    elif preset.key == "rare_event_signal":
        features, labels = _rare_event_signal(total, rng)
    elif preset.key == "overlapping_margin":
        features, labels = _overlapping_margin(total, rng)
    elif preset.key == "noisy_labels":
        features, labels = _noisy_labels(total, rng)
    elif preset.key == "sparse_interaction_signal":
        features, labels = _sparse_interaction_signal(total, rng)
    elif preset.key == "deployment_drift_probe":
        features, labels = _deployment_drift_probe(total, rng)
    elif preset.key == "active_learning_margin":
        features, labels = _active_learning_margin(total, rng)
    elif preset.key == "spurious_shortcut":
        features, labels = _spurious_shortcut(total, rng)
    elif preset.key == "subgroup_blind_spot":
        features, labels = _subgroup_blind_spot(total, rng)
    elif preset.key == "cost_sensitive_screening":
        features, labels = _cost_sensitive_screening(total, rng)
    elif preset.key == "label_audit_traps":
        features, labels = _label_audit_traps(total, rng)
    elif preset.key == "proxy_leakage_lab":
        features, labels = _proxy_leakage_lab(total, rng)
    else:
        raise ValueError(f"Unsupported preset: {preset.key}")
    return validate_dataset(features.tolist(), labels.astype(int).tolist(), min_samples=preset.min_samples, require_two_classes=True)


def save_preset_file(
    path: str | Path,
    dataset: Dataset,
    *,
    name: str,
    description: str = "",
) -> Path:
    if not name or not name.strip():
        raise DataValidationError("Preset name is required.")
    output_path = Path(path)
    payload = {
        "kind": "italtensor.dataset_preset",
        "schema_version": SCHEMA_VERSION,
        "name": name.strip(),
        "description": description.strip(),
        "training_defaults": DEFAULT_TRAINING_DEFAULTS,
        "recommended_feature_map": "linear",
        "feature_names": [f"x{index + 1}" for index in range(dataset.input_dim)],
        "label_names": {"0": "negative", "1": "positive"},
        "prediction_examples": [],
        "dataset": dataset_to_jsonable(dataset),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_preset_file(path: str | Path) -> tuple[Dataset, dict[str, Any]]:
    preset_path = Path(path)
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError("Preset file must contain a JSON object.")

    if "dataset" in payload:
        schema_version = payload.get("schema_version", payload.get("version"))
        if schema_version != SCHEMA_VERSION:
            raise DataValidationError(f"Unsupported preset schema_version: {schema_version}.")
        dataset_payload = payload["dataset"]
        metadata = {
            "name": payload.get("name", preset_path.stem),
            "description": payload.get("description", ""),
            "kind": payload.get("kind", ""),
            "schema_version": schema_version,
            "training_defaults": payload.get("training_defaults", DEFAULT_TRAINING_DEFAULTS),
            "recommended_feature_map": payload.get("recommended_feature_map"),
            "feature_names": payload.get("feature_names"),
            "label_names": payload.get("label_names"),
            "prediction_examples": payload.get("prediction_examples", []),
        }
    else:
        if "samples" not in payload:
            raise DataValidationError("Preset file must contain a dataset.")
        dataset_payload = payload
        metadata = {
            "name": preset_path.stem,
            "description": "",
            "kind": "italtensor.dataset",
            "schema_version": SCHEMA_VERSION,
            "training_defaults": DEFAULT_TRAINING_DEFAULTS,
            "recommended_feature_map": None,
            "feature_names": None,
            "label_names": {"0": "negative", "1": "positive"},
            "prediction_examples": [],
        }

    dataset = dataset_from_jsonable(dataset_payload)
    return dataset, metadata


def _linear_blobs(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    features = np.empty((total, 2), dtype=np.float32)
    features[labels == 0] = rng.normal(loc=(-1.25, -1.0), scale=0.45, size=(int(np.sum(labels == 0)), 2))
    features[labels == 1] = rng.normal(loc=(1.25, 1.0), scale=0.45, size=(int(np.sum(labels == 1)), 2))
    return _shuffle(features, labels, rng)


def _xor(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    corner_labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
    choices = np.arange(total) % 4
    rng.shuffle(choices)
    features = corners[choices] + rng.normal(0.0, 0.18, size=(total, 2)).astype(np.float32)
    labels = corner_labels[choices]
    return features, labels


def _imbalanced_blobs(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(2, round(total * 0.15))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    features = np.empty((total, 2), dtype=np.float32)
    features[:negative_count] = rng.normal(loc=(-0.35, -0.1), scale=0.65, size=(negative_count, 2))
    features[negative_count:] = rng.normal(loc=(1.15, 1.0), scale=0.35, size=(positive_count, 2))
    return _shuffle(features, labels, rng)


def _signal_plus_noise(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    informative = rng.normal(0.0, 1.0, size=(total, 2))
    score = informative[:, 0] * 1.4 - informative[:, 1] * 0.9 + rng.normal(0.0, 0.25, size=total)
    labels = (score > np.median(score)).astype(np.int32)
    noise = rng.normal(0.0, 1.0, size=(total, 4))
    features = np.concatenate([informative, noise], axis=1).astype(np.float32)
    return features, labels


def _concentric_rings(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    inner_count = int(np.sum(labels == 0))
    outer_count = int(np.sum(labels == 1))
    inner_angles = rng.uniform(0.0, 2.0 * np.pi, size=inner_count)
    outer_angles = rng.uniform(0.0, 2.0 * np.pi, size=outer_count)
    inner_radius = rng.normal(0.65, 0.07, size=inner_count)
    outer_radius = rng.normal(1.35, 0.08, size=outer_count)
    inner = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    features = np.vstack([inner, outer]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _two_moons(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    first_count = int(np.sum(labels == 0))
    second_count = int(np.sum(labels == 1))
    first_theta = rng.uniform(0.0, np.pi, size=first_count)
    second_theta = rng.uniform(0.0, np.pi, size=second_count)
    first = np.column_stack([np.cos(first_theta), np.sin(first_theta)])
    second = np.column_stack([1.0 - np.cos(second_theta), 0.45 - np.sin(second_theta)])
    features = np.vstack([first, second]) + rng.normal(0.0, 0.08, size=(total, 2))
    return _shuffle(features.astype(np.float32), labels, rng)


def _rare_event_signal(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(2, round(total * 0.08))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    negatives = rng.normal(0.0, 0.85, size=(negative_count, 4))
    positives = rng.normal((1.7, 1.5, 0.0, 0.0), (0.25, 0.25, 1.0, 1.0), size=(positive_count, 4))
    features = np.vstack([negatives, positives]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _overlapping_margin(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    covariance = np.asarray([[0.42, 0.24], [0.24, 0.38]], dtype=np.float32)
    negatives = rng.multivariate_normal(mean=(-0.55, -0.35), cov=covariance, size=negative_count)
    positives = rng.multivariate_normal(mean=(0.55, 0.35), cov=covariance, size=positive_count)
    margin_count = max(2, total // 12)
    margin_indices = rng.choice(total, size=margin_count, replace=False)
    features = np.vstack([negatives, positives]).astype(np.float32)
    features[margin_indices] = rng.normal(0.0, 0.18, size=(margin_count, 2)).astype(np.float32)
    return _shuffle(features, labels, rng)


def _noisy_labels(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _linear_blobs(total, rng)
    flip_count = max(2, int(round(total * 0.12)))
    flip_indices = rng.choice(total, size=flip_count, replace=False)
    labels = labels.copy()
    labels[flip_indices] = 1 - labels[flip_indices]
    return features.astype(np.float32), labels.astype(np.int32)


def _sparse_interaction_signal(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features = rng.normal(0.0, 1.0, size=(total, 16)).astype(np.float32)
    score = (
        1.2 * features[:, 0]
        + 0.9 * features[:, 3]
        - 1.0 * features[:, 7]
        + 1.4 * features[:, 0] * features[:, 1]
        + rng.normal(0.0, 0.35, size=total)
    )
    labels = (score > np.median(score)).astype(np.int32)
    features[:, 2] = features[:, 0] * 0.75 + rng.normal(0.0, 0.2, size=total)
    return features.astype(np.float32), labels


def _deployment_drift_probe(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    features = np.empty((total, 4), dtype=np.float32)
    features[labels == 0, :2] = rng.normal(loc=(-0.85, -0.65), scale=0.28, size=(negative_count, 2))
    features[labels == 1, :2] = rng.normal(loc=(0.85, 0.65), scale=0.28, size=(positive_count, 2))
    features[:, 2:] = rng.normal(0.0, 0.35, size=(total, 2))
    return _shuffle(features, labels, rng)


def _active_learning_margin(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    margin_count = max(4, total // 5)
    core_negative = max(2, negative_count - margin_count // 2)
    core_positive = max(2, positive_count - (margin_count - margin_count // 2))
    negative_margin = negative_count - core_negative
    positive_margin = positive_count - core_positive

    negatives = rng.normal(loc=(-1.1, -0.45), scale=(0.32, 0.28), size=(core_negative, 2))
    positives = rng.normal(loc=(1.1, 0.45), scale=(0.32, 0.28), size=(core_positive, 2))
    margin_negatives = rng.normal(loc=(-0.12, 0.0), scale=(0.18, 0.32), size=(negative_margin, 2))
    margin_positives = rng.normal(loc=(0.12, 0.0), scale=(0.18, 0.32), size=(positive_margin, 2))
    features = np.vstack([negatives, margin_negatives, positives, margin_positives]).astype(np.float32)
    labels = np.asarray(
        [0] * (core_negative + negative_margin) + [1] * (core_positive + positive_margin),
        dtype=np.int32,
    )
    return _shuffle(features, labels, rng)


def _spurious_shortcut(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    signed = np.where(labels == 1, 1.0, -1.0)
    stable_signal = signed * 0.55 + rng.normal(0.0, 0.55, size=total)
    context_noise = rng.normal(0.0, 1.0, size=total)
    shortcut_signal = signed * 1.2 + rng.normal(0.0, 0.12, size=total)
    conflict_count = max(2, total // 12)
    conflict_indices = rng.choice(total, size=conflict_count, replace=False)
    stable_signal[conflict_indices] *= -1.0
    features = np.column_stack([stable_signal, context_noise, shortcut_signal]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _subgroup_blind_spot(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    subgroup = rng.random(total) < 0.22
    primary_signal = rng.normal(0.0, 1.0, size=total)
    context_noise = rng.normal(0.0, 1.0, size=total)
    majority_score = primary_signal + rng.normal(0.0, 0.25, size=total)
    minority_score = -primary_signal + 0.25 * context_noise + rng.normal(0.0, 0.25, size=total)
    score = np.where(subgroup, minority_score, majority_score)
    labels = (score > 0.0).astype(np.int32)
    subgroup_marker = subgroup.astype(np.float32) + rng.normal(0.0, 0.04, size=total)
    features = np.column_stack([primary_signal, subgroup_marker, context_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _cost_sensitive_screening(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(3, round(total * 0.12))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    negatives = rng.normal(loc=(-0.25, -0.15, 0.0), scale=(0.55, 0.45, 1.0), size=(negative_count, 3))
    positives = rng.normal(loc=(0.65, 0.45, 0.0), scale=(0.45, 0.4, 1.0), size=(positive_count, 3))
    features = np.vstack([negatives, positives]).astype(np.float32)
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    return _shuffle(features, labels, rng)


def _label_audit_traps(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _linear_blobs(total, rng)
    flip_count = max(2, int(round(total * 0.08)))
    negative_indices = np.where(labels == 0)[0]
    positive_indices = np.where(labels == 1)[0]
    chosen_negatives = rng.choice(negative_indices, size=flip_count // 2, replace=False)
    chosen_positives = rng.choice(positive_indices, size=flip_count - chosen_negatives.shape[0], replace=False)
    labels = labels.copy()
    labels[chosen_negatives] = 1
    labels[chosen_positives] = 0
    return features.astype(np.float32), labels.astype(np.int32)


def _proxy_leakage_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    real_signal = rng.normal(0.0, 1.0, size=total)
    weak_signal = rng.normal(0.0, 1.0, size=total)
    latent = 0.9 * real_signal + 0.35 * weak_signal + rng.normal(0.0, 0.35, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    signed = np.where(labels == 1, 1.0, -1.0)
    proxy_code = signed * 1.35 + rng.normal(0.0, 0.08, size=total)
    conflict_count = max(2, total // 12)
    conflict_indices = rng.choice(total, size=conflict_count, replace=False)
    proxy_code[conflict_indices] *= -1.0
    background_noise = rng.normal(0.0, 1.0, size=total)
    features = np.column_stack([real_signal, weak_signal, proxy_code, background_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _balanced_labels(total: int) -> np.ndarray:
    labels = np.asarray([0] * (total // 2) + [1] * (total - total // 2), dtype=np.int32)
    return labels


def _shuffle(features: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.permutation(labels.shape[0])
    return features[indices].astype(np.float32), labels[indices].astype(np.int32)


def _metadata_from_preset(preset: PresetInfo) -> dict[str, object]:
    return {
        "name": preset.name,
        "description": preset.description,
        "schema_version": SCHEMA_VERSION,
        "input_dim": preset.input_dim,
        "recommended_feature_map": preset.recommended_feature_map,
        "feature_names": list(preset.feature_names),
        "label_names": {"0": preset.label_names[0], "1": preset.label_names[1]},
        "training_defaults": dict(preset.training_defaults),
        "prediction_examples": [dict(example) for example in preset.prediction_examples],
    }
