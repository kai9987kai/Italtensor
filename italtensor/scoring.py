from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import DataValidationError
from .experiments import conformal_label_set
from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


@dataclass(frozen=True)
class PredictionTable:
    feature_names: list[str]
    features: np.ndarray


def load_prediction_csv(path: str | Path, expected_dim: int | None = None) -> PredictionTable:
    rows: list[list[str]] = []
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if row and any(cell.strip() for cell in row):
                rows.append(row)

    if not rows:
        raise DataValidationError("Prediction CSV file is empty.")

    feature_names: list[str] | None = None
    data_rows = rows
    if _looks_like_prediction_header(rows[0]):
        feature_names = [cell.strip() or f"x{index + 1}" for index, cell in enumerate(rows[0])]
        data_rows = rows[1:]

    if not data_rows:
        raise DataValidationError("Prediction CSV file has no data rows.")

    parsed: list[list[float]] = []
    inferred_dim = expected_dim
    for row_number, row in enumerate(data_rows, start=2 if feature_names is not None else 1):
        try:
            parsed_row = [_parse_prediction_float(cell) for cell in row]
            if inferred_dim is None:
                inferred_dim = len(parsed_row)
            if len(parsed_row) != inferred_dim:
                raise DataValidationError(f"Expected {inferred_dim} features, got {len(parsed_row)}.")
        except DataValidationError as exc:
            raise DataValidationError(f"Prediction CSV row {row_number}: {exc}") from exc
        parsed.append(parsed_row)

    input_dim = inferred_dim or 0
    if input_dim <= 0:
        raise DataValidationError("Prediction CSV must contain at least one feature column.")
    if feature_names is None:
        feature_names = [f"x{index + 1}" for index in range(input_dim)]
    if len(feature_names) != input_dim:
        raise DataValidationError(f"Prediction CSV header has {len(feature_names)} columns, expected {input_dim}.")

    return PredictionTable(feature_names=feature_names, features=np.asarray(parsed, dtype=np.float32))


def score_prediction_rows(
    model: Any,
    features: np.ndarray,
    *,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    uncertainty_metadata: dict[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    raw_features = np.asarray(features, dtype=np.float32)
    if raw_features.ndim != 2:
        raise DataValidationError("Prediction features must be a 2D array.")
    prepared = preprocessor.transform(raw_features) if preprocessor is not None else raw_features
    probabilities = predict_probability(model, prepared)
    drift = _drift_diagnostics(prepared if preprocessor is not None else None, raw_features.shape[0])
    quantile = (uncertainty_metadata or {}).get("conformal_quantile")
    scored: list[dict[str, float | int | str]] = []
    for probability, drift_data in zip(probabilities, drift, strict=True):
        probability_value = float(probability)
        label = int(probability_value >= threshold)
        if quantile is None:
            conformal_set = "unavailable"
        else:
            conformal_set = _format_label_set(conformal_label_set(probability_value, float(quantile)))
        uncertainty_score = _uncertainty_score(probability_value, threshold, conformal_set)
        ood_flag = drift_data["ood_flag"] == 1
        scored.append(
            {
                "probability": probability_value,
                "label": label,
                "conformal_set": conformal_set,
                "uncertainty_score": uncertainty_score,
                "drift_score": drift_data["drift_score"],
                "max_abs_z": drift_data["max_abs_z"],
                "ood_flag": drift_data["ood_flag"],
                "review_priority": _review_priority(uncertainty_score, conformal_set, ood_flag),
            }
        )
    return scored


def export_prediction_csv(
    output_path: str | Path,
    table: PredictionTable,
    scored_rows: list[dict[str, float | int | str]],
) -> Path:
    if table.features.shape[0] != len(scored_rows):
        raise DataValidationError("Prediction row count does not match score count.")
    path = Path(output_path)
    header = table.feature_names + [
        "italtensor_probability",
        "italtensor_label",
        "italtensor_conformal_set",
        "italtensor_uncertainty_score",
        "italtensor_drift_score",
        "italtensor_max_abs_z",
        "italtensor_ood_flag",
        "italtensor_review_priority",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for features, score in zip(table.features, scored_rows, strict=True):
            writer.writerow(
                [float(value) for value in features]
                + [
                    f"{float(score['probability']):.8f}",
                    int(score["label"]),
                    score["conformal_set"],
                    f"{float(score['uncertainty_score']):.8f}",
                    _format_optional_float(score["drift_score"]),
                    _format_optional_float(score["max_abs_z"]),
                    score["ood_flag"],
                    score["review_priority"],
                ]
            )
    return path


def score_prediction_csv(
    model: Any,
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected_dim: int | None = None,
    preprocessor: FeatureStandardizer | None = None,
    threshold: float = 0.5,
    uncertainty_metadata: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    table = load_prediction_csv(input_path, expected_dim)
    scored = score_prediction_rows(
        model,
        table.features,
        preprocessor=preprocessor,
        threshold=threshold,
        uncertainty_metadata=uncertainty_metadata,
    )
    return export_prediction_csv(output_path, table, scored), len(scored)


def _parse_prediction_float(cell: str) -> float:
    try:
        number = float(cell.strip())
    except ValueError as exc:
        raise DataValidationError("All prediction feature columns must be numeric.") from exc
    if not math.isfinite(number):
        raise DataValidationError("All prediction feature columns must be finite.")
    return number


def _looks_like_prediction_header(row: list[str]) -> bool:
    for cell in row:
        try:
            float(cell.strip())
        except ValueError:
            continue
        return False
    return True


def _format_label_set(label_set: list[int]) -> str:
    if not label_set:
        return "abstain"
    if len(label_set) == 2:
        return "{0,1}"
    return "{" + str(label_set[0]) + "}"


def _uncertainty_score(probability: float, threshold: float, conformal_set: str) -> float:
    normalized_margin = abs(probability - threshold) / max(threshold, 1.0 - threshold, 1e-6)
    score = 1.0 - min(max(normalized_margin, 0.0), 1.0)
    if conformal_set in {"{0,1}", "unavailable"}:
        return float(score)
    return float(max(score, 0.0))


def _drift_diagnostics(
    prepared_features: np.ndarray | None,
    row_count: int,
) -> list[dict[str, float | int | str]]:
    if prepared_features is None:
        return [
            {"drift_score": "unavailable", "max_abs_z": "unavailable", "ood_flag": "unavailable"}
            for _ in range(row_count)
        ]
    values = np.asarray(prepared_features, dtype=np.float32)
    if values.ndim != 2:
        raise DataValidationError("Prepared prediction features must be a 2D array.")
    rms_z = np.sqrt(np.mean(values * values, axis=1))
    max_abs_z = np.max(np.abs(values), axis=1)
    return [
        {
            "drift_score": float(rms),
            "max_abs_z": float(maximum),
            "ood_flag": 1 if float(maximum) >= 3.0 else 0,
        }
        for rms, maximum in zip(rms_z, max_abs_z, strict=True)
    ]


def _format_optional_float(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.8f}"


def _review_priority(uncertainty_score: float, conformal_set: str, drift_flag: bool) -> str:
    if drift_flag or conformal_set in {"{0,1}", "abstain"} or uncertainty_score >= 0.8:
        return "high"
    if uncertainty_score >= 0.6:
        return "medium"
    return "low"
