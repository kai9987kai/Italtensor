from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Dataset, DataValidationError, dataset_from_jsonable, dataset_to_jsonable, validate_dataset

SCHEMA_VERSION = 1
DEFAULT_TRAINING_DEFAULTS = {"epochs": 50, "batch_size": 16, "trials": 8}


@dataclass(frozen=True)
class PresetInfo:
    key: str
    name: str
    description: str
    default_samples: int
    min_samples: int = 4


BUILT_IN_PRESETS: tuple[PresetInfo, ...] = (
    PresetInfo(
        key="linear_blobs",
        name="Linear blobs",
        description="Two separable 2D Gaussian clusters for quick sanity checks.",
        default_samples=80,
    ),
    PresetInfo(
        key="xor",
        name="XOR pattern",
        description="A nonlinear 2D pattern that rewards hidden layers over a linear boundary.",
        default_samples=96,
    ),
    PresetInfo(
        key="imbalanced_blobs",
        name="Imbalanced blobs",
        description="A skewed 2D binary dataset for testing class weights and balanced metrics.",
        default_samples=100,
    ),
    PresetInfo(
        key="signal_plus_noise",
        name="Signal plus noise",
        description="Six features where only the first two drive the label, useful for feature importance.",
        default_samples=120,
    ),
    PresetInfo(
        key="concentric_rings",
        name="Concentric rings",
        description="A radial nonlinear dataset for trying RFF feature maps.",
        default_samples=120,
    ),
    PresetInfo(
        key="two_moons",
        name="Two moons",
        description="Interleaving crescent shapes for nonlinear boundary experiments.",
        default_samples=120,
    ),
    PresetInfo(
        key="rare_event_signal",
        name="Rare event signal",
        description="A heavily imbalanced dataset with a compact positive region.",
        default_samples=160,
    ),
)


def preset_labels() -> list[str]:
    return [preset.name for preset in BUILT_IN_PRESETS]


def preset_by_name(name: str) -> PresetInfo:
    for preset in BUILT_IN_PRESETS:
        if preset.name == name or preset.key == name:
            return preset
    raise ValueError(f"Unknown preset: {name}")


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


def _balanced_labels(total: int) -> np.ndarray:
    labels = np.asarray([0] * (total // 2) + [1] * (total - total // 2), dtype=np.int32)
    return labels


def _shuffle(features: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.permutation(labels.shape[0])
    return features[indices].astype(np.float32), labels[indices].astype(np.int32)
