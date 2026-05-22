from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class DataValidationError(ValueError):
    """Raised when user-provided training or prediction data is invalid."""


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    input_dim: int

    @property
    def sample_count(self) -> int:
        return int(self.labels.shape[0])


def parse_training_example(raw_text: str, expected_dim: int | None = None) -> tuple[list[float], int]:
    """Parse one JSON training example: [[feature, ...], label]."""
    value = _load_json(raw_text, "training example")
    if not isinstance(value, list) or len(value) != 2:
        raise DataValidationError("Training input must be JSON like [[0.1, 0.2], 1].")

    features = _parse_feature_vector(value[0], expected_dim)
    label = _parse_label(value[1])
    return features, label


def parse_prediction_vector(raw_text: str, expected_dim: int | None = None) -> list[float]:
    """Parse one JSON prediction vector: [feature, ...]."""
    value = _load_json(raw_text, "prediction vector")
    return _parse_feature_vector(value, expected_dim)


def validate_dataset(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    min_samples: int = 1,
    require_two_classes: bool = False,
) -> Dataset:
    if len(features) != len(labels):
        raise DataValidationError("Feature and label counts do not match.")
    if len(features) < min_samples:
        raise DataValidationError(f"Dataset needs at least {min_samples} sample(s).")

    expected_dim: int | None = None
    parsed_features: list[list[float]] = []
    parsed_labels: list[int] = []

    for row_index, row in enumerate(features, start=1):
        try:
            parsed_row = _parse_feature_vector(row, expected_dim)
        except DataValidationError as exc:
            raise DataValidationError(f"Sample {row_index}: {exc}") from exc
        if expected_dim is None:
            expected_dim = len(parsed_row)
        parsed_features.append(parsed_row)

    for row_index, label in enumerate(labels, start=1):
        try:
            parsed_labels.append(_parse_label(label))
        except DataValidationError as exc:
            raise DataValidationError(f"Label {row_index}: {exc}") from exc

    if require_two_classes and len(set(parsed_labels)) < 2:
        raise DataValidationError("Dataset must contain both labels 0 and 1.")

    feature_array = np.asarray(parsed_features, dtype=np.float32)
    label_array = np.asarray(parsed_labels, dtype=np.int32)
    return Dataset(feature_array, label_array, expected_dim or 0)


def load_csv_dataset(path: str | Path, expected_dim: int | None = None) -> Dataset:
    """Load a CSV dataset where the last column is the binary label."""
    rows: list[list[str]] = []
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if row and any(cell.strip() for cell in row):
                rows.append(row)

    if not rows:
        raise DataValidationError("CSV file is empty.")

    parsed_rows = _parse_csv_rows(rows, expected_dim)
    features = [features for features, _ in parsed_rows]
    labels = [label for _, label in parsed_rows]
    return validate_dataset(features, labels, min_samples=1)


def dataset_to_jsonable(dataset: Dataset) -> dict[str, object]:
    return {
        "input_dim": dataset.input_dim,
        "samples": [
            {"features": features.tolist(), "label": int(label)}
            for features, label in zip(dataset.features, dataset.labels, strict=True)
        ],
    }


def dataset_from_jsonable(value: object) -> Dataset:
    if not isinstance(value, dict):
        raise DataValidationError("Dataset file must contain a JSON object.")
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise DataValidationError("Dataset file must contain a samples list.")

    features: list[list[float]] = []
    labels: list[int] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise DataValidationError(f"Sample {index} must be an object.")
        features.append(_parse_feature_vector(sample.get("features"), None))
        labels.append(_parse_label(sample.get("label")))

    dataset = validate_dataset(features, labels, min_samples=1)
    expected_dim = value.get("input_dim")
    if expected_dim is not None and int(expected_dim) != dataset.input_dim:
        raise DataValidationError("Saved input_dim does not match the samples.")
    return dataset


def _load_json(raw_text: str, description: str) -> object:
    if not raw_text or not raw_text.strip():
        raise DataValidationError(f"Enter a {description}.")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid JSON {description}.") from exc


def _parse_feature_vector(value: object, expected_dim: int | None) -> list[float]:
    if not isinstance(value, list):
        raise DataValidationError("Feature vector must be a JSON array.")
    if not value:
        raise DataValidationError("Feature vector cannot be empty.")

    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DataValidationError("All features must be numbers.")
        number = float(item)
        if not math.isfinite(number):
            raise DataValidationError("All features must be finite numbers.")
        parsed.append(number)

    if expected_dim is not None and len(parsed) != expected_dim:
        raise DataValidationError(f"Expected {expected_dim} features, got {len(parsed)}.")
    return parsed


def _parse_label(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise DataValidationError("Label must be integer 0 or 1.")
    return int(value)


def _parse_csv_rows(
    rows: Iterable[Sequence[str]], expected_dim: int | None
) -> list[tuple[list[float], int]]:
    parsed: list[tuple[list[float], int]] = []
    inferred_dim = expected_dim

    for row_number, row in enumerate(rows, start=1):
        try:
            if len(row) < 2:
                raise DataValidationError("Each CSV row needs at least one feature and one label.")
            features = _parse_feature_vector([_parse_csv_float(cell) for cell in row[:-1]], inferred_dim)
            label = _parse_label(_parse_csv_label(row[-1]))
        except DataValidationError as exc:
            if row_number == 1 and _looks_like_header(row):
                continue
            raise DataValidationError(f"CSV row {row_number}: {exc}") from exc

        if inferred_dim is None:
            inferred_dim = len(features)
        parsed.append((features, label))

    if not parsed:
        raise DataValidationError("CSV file has no data rows.")
    return parsed


def _parse_csv_float(cell: str) -> float:
    try:
        number = float(cell.strip())
    except ValueError as exc:
        raise DataValidationError("All feature columns must be numeric.") from exc
    if not math.isfinite(number):
        raise DataValidationError("All feature columns must be finite.")
    return number


def _parse_csv_label(cell: str) -> int:
    stripped = cell.strip()
    if stripped not in {"0", "1"}:
        raise DataValidationError("Final CSV column must be label 0 or 1.")
    return int(stripped)


def _looks_like_header(row: Sequence[str]) -> bool:
    for cell in row:
        try:
            float(cell.strip())
        except ValueError:
            continue
        return False
    return True
