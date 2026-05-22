from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: tuple[int, ...] = (32,)
    learning_rate: float = 0.001
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 5
    random_seed: int = 42

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ModelConfig":
        return cls(
            hidden_layers=tuple(int(unit) for unit in value.get("hidden_layers", [32])),
            learning_rate=float(value.get("learning_rate", 0.001)),
            batch_size=int(value.get("batch_size", 16)),
            max_epochs=int(value.get("max_epochs", 50)),
            patience=int(value.get("patience", 5)),
            random_seed=int(value.get("random_seed", 42)),
        )


def build_model(input_dim: int, config: ModelConfig):
    if input_dim <= 0:
        raise ValueError("input_dim must be greater than zero.")
    if not config.hidden_layers:
        raise ValueError("At least one hidden layer is required.")

    tf = _tensorflow()
    tf.keras.utils.set_random_seed(config.random_seed)

    layers: list[Any] = [tf.keras.Input(shape=(input_dim,))]
    for units in config.hidden_layers:
        if units <= 0:
            raise ValueError("Hidden layer sizes must be greater than zero.")
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: int = 0,
):
    tf = _tensorflow()
    callbacks = []
    if validation_data is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        )

    model = build_model(int(features.shape[1]), config)
    history = model.fit(
        features,
        labels,
        validation_data=validation_data,
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, {name: [float(item) for item in values] for name, values in history.history.items()}


def predict_probability(model: Any, samples: Sequence[float] | np.ndarray) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.float32)
    if sample_array.ndim == 1:
        sample_array = sample_array.reshape(1, -1)
    if sample_array.ndim != 2:
        raise ValueError("Prediction input must be one vector or a 2D feature array.")
    predictions = model.predict(sample_array, verbose=0)
    return np.asarray(predictions, dtype=np.float32).reshape(-1)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return tf
