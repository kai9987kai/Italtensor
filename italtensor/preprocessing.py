from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Features must be a 2D array.")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    @classmethod
    def identity(cls, input_dim: int) -> "FeatureStandardizer":
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")
        return cls(mean=np.zeros(input_dim, dtype=np.float32), scale=np.ones(input_dim, dtype=np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("Features must be one vector or a 2D array.")
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Expected {self.mean.shape[0]} features, got {values.shape[1]}.")
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "standardize",
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None, input_dim: int | None = None) -> "FeatureStandardizer | None":
        if not value:
            return cls.identity(input_dim) if input_dim else None
        method = value.get("method")
        if method != "standardize":
            raise ValueError(f"Unsupported preprocessing method: {method}")
        mean = np.asarray(value.get("mean"), dtype=np.float32)
        scale = np.asarray(value.get("scale"), dtype=np.float32)
        if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
            raise ValueError("Invalid preprocessing metadata.")
        if input_dim is not None and mean.shape[0] != input_dim:
            raise ValueError(f"Preprocessing metadata expects {mean.shape[0]} features, model expects {input_dim}.")
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)
