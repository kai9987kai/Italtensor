"""Explicit backend resolution, multi-model run queue, and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .experiments import ExperimentResult, train_single_model
from .modeling import ModelConfig, train_model

BACKEND_AUTO = "auto"
BACKEND_NUMPY = "numpy"
BACKEND_KERAS = "keras"
BACKEND_CHOICES = (BACKEND_AUTO, BACKEND_NUMPY, BACKEND_KERAS)


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def available_backends() -> list[str]:
    """Backends that can actually run in the current environment."""
    backends = [BACKEND_NUMPY]
    if tensorflow_available():
        backends.append(BACKEND_KERAS)
    return backends


def resolve_backend(preference: str = BACKEND_AUTO) -> str:
    """Resolve user preference to a concrete backend id (numpy or keras)."""
    pref = (preference or BACKEND_AUTO).lower().strip()
    if pref not in BACKEND_CHOICES:
        raise ValueError(f"backend must be one of: {', '.join(BACKEND_CHOICES)}.")
    if pref == BACKEND_NUMPY:
        return BACKEND_NUMPY
    if pref == BACKEND_KERAS:
        if not tensorflow_available():
            raise RuntimeError(
                "Keras backend requested but TensorFlow is not installed. "
                "Install with: python -m pip install -r requirements-tensorflow.txt"
            )
        return BACKEND_KERAS
    return BACKEND_KERAS if tensorflow_available() else BACKEND_NUMPY


def train_with_backend(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    class_weight: dict[int, float] | None = None,
    verbose: int = 0,
    backend: str | None = None,
):
    """Train using an explicitly resolved backend."""
    resolved = resolve_backend(backend or getattr(config, "backend", BACKEND_AUTO))
    if resolved == BACKEND_NUMPY:
        from .modeling import train_numpy_model

        return train_numpy_model(
            features,
            labels,
            config,
            validation_data=validation_data,
            class_weight=class_weight,
        )
    return train_model(
        features,
        labels,
        config,
        validation_data=validation_data,
        class_weight=class_weight,
        verbose=verbose,
        force_backend=BACKEND_KERAS,
    )


@dataclass(frozen=True)
class ModelRunSpec:
    """One queued training job."""

    name: str
    config: ModelConfig
    backend: str = BACKEND_AUTO


@dataclass
class ModelRunQueue:
    """Ordered list of training jobs for multi-model sweeps."""

    specs: list[ModelRunSpec] = field(default_factory=list)

    def add(self, spec: ModelRunSpec) -> None:
        self.specs.append(spec)

    @classmethod
    def multi_backend_sweep(
        cls,
        base_config: ModelConfig,
        *,
        include_numpy: bool = True,
        include_keras: bool | None = None,
    ) -> ModelRunQueue:
        """Build a sweep across available backends with the same hyperparameters."""
        queue = cls()
        if include_numpy:
            queue.add(
                ModelRunSpec(
                    name="numpy-logistic",
                    config=_config_with_backend(base_config, BACKEND_NUMPY),
                    backend=BACKEND_NUMPY,
                )
            )
        if include_keras is None:
            include_keras = tensorflow_available()
        if include_keras and tensorflow_available():
            queue.add(
                ModelRunSpec(
                    name="keras-mlp",
                    config=_config_with_backend(base_config, BACKEND_KERAS),
                    backend=BACKEND_KERAS,
                )
            )
        if not queue.specs:
            raise RuntimeError("No backends available for a multi-backend sweep.")
        return queue


def _config_with_backend(config: ModelConfig, backend: str) -> ModelConfig:
    data = config.to_dict()
    data["backend"] = backend
    return ModelConfig.from_dict(data)


def run_model_queue(
    features: np.ndarray,
    labels: np.ndarray,
    queue: ModelRunQueue,
    *,
    progress_callback: Callable[[int, int, ExperimentResult], None] | None = None,
) -> list[ExperimentResult]:
    """Run each spec in order and return all experiment results."""
    results: list[ExperimentResult] = []
    total = len(queue.specs)
    for index, spec in enumerate(queue.specs, start=1):
        cfg = _config_with_backend(spec.config, spec.backend)
        result = train_single_model(features, labels, cfg)
        results.append(result)
        if progress_callback is not None:
            progress_callback(index, total, result)
    return results


def select_best_from_runs(results: list[ExperimentResult]) -> ExperimentResult:
    """Pick the best run by validation F1, then accuracy, then lower loss."""
    if not results:
        raise ValueError("No model runs to select from.")

    def sort_key(item: ExperimentResult) -> tuple[float, float, float]:
        metrics = item.metrics
        f1 = float(metrics.get("f1", 0.0))
        acc = float(metrics.get("accuracy", 0.0))
        loss = float(metrics.get("validation_loss", float("inf")))
        return (f1, acc, -loss)

    return max(results, key=sort_key)
