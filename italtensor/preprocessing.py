from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def compute_mutual_information(features: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Compute Mutual Information between each feature and binary labels using binning."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    n_samples, n_features = x.shape
    mi_scores = np.zeros(n_features)
    
    p_y = np.array([np.mean(y == 0), np.mean(y == 1)])
    p_y = np.maximum(p_y, 1e-12)
    
    for j in range(n_features):
        feature_col = x[:, j]
        if np.all(feature_col == feature_col[0]):
            mi_scores[j] = 0.0
            continue
            
        bins = np.linspace(np.min(feature_col), np.max(feature_col), n_bins + 1)
        binned = np.digitize(feature_col, bins[:-1]) - 1
        
        joint_counts = np.zeros((n_bins, 2))
        for val, lbl in zip(binned, y):
            val = min(max(val, 0), n_bins - 1)
            joint_counts[val, lbl] += 1
            
        p_xy = joint_counts / n_samples
        p_x = np.sum(p_xy, axis=1)
        
        mi = 0.0
        for xi in range(n_bins):
            for yi in [0, 1]:
                pxy = p_xy[xi, yi]
                px = p_x[xi]
                py = p_y[yi]
                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * np.log2(pxy / (px * py))
        mi_scores[j] = mi
        
    return mi_scores


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    selected_indices: list[int] | None = None

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
    def fit_with_selection(cls, features: np.ndarray, labels: np.ndarray, k: int) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Features must be a 2D array.")
        n_features = values.shape[1]
        k = min(max(1, k), n_features)
        
        mi_scores = compute_mutual_information(values, labels)
        selected_indices = np.argsort(mi_scores)[::-1][:k].tolist()
        selected_indices.sort()
        
        selected_features = values[:, selected_indices]
        mean = selected_features.mean(axis=0)
        scale = selected_features.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            selected_indices=selected_indices,
        )

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
            
        if self.selected_indices is not None:
            values = values[:, self.selected_indices]
            
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Expected {self.mean.shape[0]} features, got {values.shape[1]}.")
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "standardize",
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "selected_indices": self.selected_indices,
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
            
        selected_indices = value.get("selected_indices")
        if selected_indices is not None:
            selected_indices = [int(i) for i in selected_indices]
            if input_dim is not None:
                max_idx = max(selected_indices) if selected_indices else 0
                if max_idx >= input_dim:
                    raise ValueError(f"Feature selection index {max_idx} is out of bounds for input dimension {input_dim}.")
        else:
            if input_dim is not None and mean.shape[0] != input_dim:
                raise ValueError(f"Preprocessing metadata expects {mean.shape[0]} features, model expects {input_dim}.")
            
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale, selected_indices=selected_indices)
