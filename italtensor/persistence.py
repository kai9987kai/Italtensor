from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import Dataset, dataset_from_jsonable, dataset_to_jsonable
from .modeling import ModelConfig
from .preprocessing import FeatureStandardizer


def save_dataset(path: str | Path, dataset: Dataset) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(dataset_to_jsonable(dataset), indent=2), encoding="utf-8")
    return output_path


def load_dataset(path: str | Path) -> Dataset:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return dataset_from_jsonable(value)


def save_model_bundle(
    model: Any,
    path: str | Path,
    *,
    input_dim: int,
    config: ModelConfig,
    metrics: dict[str, float | int] | None = None,
    threshold: float = 0.5,
    preprocessor: FeatureStandardizer | None = None,
    feature_importances: list[dict[str, float | int]] | None = None,
) -> tuple[Path, Path]:
    model_path = Path(path)
    if model_path.suffix != ".keras":
        model_path = model_path.with_suffix(".keras")

    resolved_preprocessor = preprocessor or FeatureStandardizer.identity(input_dim)
    if resolved_preprocessor.mean.shape[0] != input_dim:
        raise ValueError(
            f"Preprocessing metadata expects {resolved_preprocessor.mean.shape[0]} features, "
            f"model expects {input_dim}."
        )

    model.save(str(model_path))
    metadata_path = model_metadata_path(model_path)
    metadata = {
        "input_dim": input_dim,
        "label_schema": {"negative": 0, "positive": 1},
        "best_config": config.to_dict(),
        "validation_metrics": metrics or {},
        "threshold": float(threshold),
        "preprocessing": resolved_preprocessor.to_dict(),
        "feature_importances": feature_importances or [],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_model_bundle(path: str | Path):
    model_path = Path(path)
    tf = _tensorflow()
    model = tf.keras.models.load_model(str(model_path))

    metadata_path = model_metadata_path(model_path)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def model_metadata_path(model_path: str | Path) -> Path:
    return Path(str(model_path) + ".json")


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return tf
