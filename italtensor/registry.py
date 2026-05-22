"""In-session model registry slot types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .modeling import ModelConfig
from .preprocessing import FeatureStandardizer


@dataclass
class ModelSlot:
    model: Any
    config: ModelConfig
    metrics: dict[str, float | int]
    preprocessor: FeatureStandardizer | None
    threshold: float
    name: str
