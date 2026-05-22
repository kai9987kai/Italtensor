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
    feature_map: str = "linear"
    rff_components: int = 64
    rff_gamma: float = 1.0
    l1_penalty: float = 0.0
    feature_selection_k: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "random_seed": self.random_seed,
            "feature_map": self.feature_map,
            "rff_components": self.rff_components,
            "rff_gamma": self.rff_gamma,
            "l1_penalty": self.l1_penalty,
            "feature_selection_k": self.feature_selection_k,
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
            feature_map=str(value.get("feature_map", "linear")),
            rff_components=int(value.get("rff_components", 64)),
            rff_gamma=float(value.get("rff_gamma", 1.0)),
            l1_penalty=float(value.get("l1_penalty", 0.0)),
            feature_selection_k=value.get("feature_selection_k") if value.get("feature_selection_k") is None else int(value.get("feature_selection_k")),
        )


@dataclass
class NumpyBinaryClassifier:
    weights: np.ndarray
    bias: float
    raw_input_dim: int | None = None
    feature_map: str = "linear"
    rff_weights: np.ndarray | None = None
    rff_bias: np.ndarray | None = None
    backend: str = "numpy-logistic"
    calibration_a: float = 1.0
    calibration_b: float = 0.0

    @property
    def input_dim(self) -> int:
        return int(self.raw_input_dim or self.weights.shape[0])

    @property
    def input_shape(self) -> tuple[None, int]:
        return (None, self.input_dim)

    def predict(self, samples: Sequence[float] | np.ndarray, verbose: int = 0) -> np.ndarray:
        sample_array = np.asarray(samples, dtype=np.float32)
        if sample_array.ndim == 1:
            sample_array = sample_array.reshape(1, -1)
        if sample_array.ndim != 2:
            raise ValueError("Prediction input must be one vector or a 2D feature array.")
        if sample_array.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {sample_array.shape[1]}.")
        mapped = _apply_feature_map(sample_array, self.feature_map, self.rff_weights, self.rff_bias)
        if mapped.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Model weights expect {self.weights.shape[0]} mapped features, got {mapped.shape[1]}.")
        logits = mapped @ self.weights + self.bias
        if self.calibration_a != 1.0 or self.calibration_b != 0.0:
            calibrated_logits = self.calibration_a * logits + self.calibration_b
            return _sigmoid(calibrated_logits).reshape(-1, 1).astype(np.float32)
        return _sigmoid(logits).reshape(-1, 1).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_format_version": 1,
            "backend": self.backend,
            "input_dim": self.input_dim,
            "feature_map": self.feature_map,
            "rff_weights": self.rff_weights.astype(float).tolist() if self.rff_weights is not None else None,
            "rff_bias": self.rff_bias.astype(float).tolist() if self.rff_bias is not None else None,
            "weights": self.weights.astype(float).tolist(),
            "bias": float(self.bias),
            "calibration_a": float(self.calibration_a),
            "calibration_b": float(self.calibration_b),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "NumpyBinaryClassifier":
        weights = np.asarray(value.get("weights"), dtype=np.float32)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("Invalid NumPy model weights.")
        input_dim = value.get("input_dim")
        if input_dim is not None and int(input_dim) != int(weights.shape[0]):
            feature_map = str(value.get("feature_map", "linear"))
            if feature_map == "linear":
                raise ValueError("NumPy model input_dim does not match its weights.")
        else:
            feature_map = str(value.get("feature_map", "linear"))
        rff_weights = value.get("rff_weights")
        rff_bias = value.get("rff_bias")
        parsed_rff_weights = np.asarray(rff_weights, dtype=np.float32) if rff_weights is not None else None
        parsed_rff_bias = np.asarray(rff_bias, dtype=np.float32) if rff_bias is not None else None
        if feature_map == "rff":
            if parsed_rff_weights is None or parsed_rff_bias is None:
                raise ValueError("RFF model is missing random feature parameters.")
            if parsed_rff_weights.ndim != 2 or parsed_rff_bias.ndim != 1:
                raise ValueError("Invalid RFF parameter shapes.")
            if input_dim is not None and parsed_rff_weights.shape[0] != int(input_dim):
                raise ValueError("RFF input_dim does not match its random weights.")
            if parsed_rff_bias.shape[0] != parsed_rff_weights.shape[1]:
                raise ValueError("RFF bias length does not match random weights.")
            if weights.shape[0] != parsed_rff_weights.shape[1]:
                raise ValueError("Classifier weights do not match RFF mapped dimension.")
        elif weights.shape[0] != int(input_dim or weights.shape[0]):
            raise ValueError("Classifier weights do not match input dimension.")
        return cls(
            weights=weights,
            bias=float(value.get("bias", 0.0)),
            raw_input_dim=int(input_dim) if input_dim is not None else None,
            feature_map=feature_map,
            rff_weights=parsed_rff_weights,
            rff_bias=parsed_rff_bias,
            calibration_a=float(value.get("calibration_a", 1.0)),
            calibration_b=float(value.get("calibration_b", 0.0)),
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
    class_weight: dict[int, float] | None = None,
    verbose: int = 0,
):
    tf = _try_tensorflow()
    if tf is None:
        return train_numpy_model(
            features,
            labels,
            config,
            validation_data=validation_data,
            class_weight=class_weight,
        )

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
        class_weight=class_weight,
        verbose=verbose,
    )
    return model, {name: [float(item) for item in values] for name, values in history.history.items()}


def train_numpy_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    class_weight: dict[int, float] | None = None,
) -> tuple[NumpyBinaryClassifier, dict[str, list[float]]]:
    x_train = np.asarray(features, dtype=np.float32)
    y_train = np.asarray(labels, dtype=np.float32).reshape(-1)
    if x_train.ndim != 2:
        raise ValueError("Training features must be a 2D array.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training feature and label counts do not match.")

    rng = np.random.default_rng(config.random_seed)
    feature_map = _build_feature_map(x_train.shape[1], config, rng)
    x_train_mapped = feature_map.transform(x_train)
    weights = rng.normal(0.0, 0.05, size=x_train_mapped.shape[1]).astype(np.float32)
    bias = 0.0
    sample_weights = _sample_weights(y_train, class_weight)
    learning_rate = min(max(config.learning_rate, 1e-4), 0.1)
    l2 = 1e-4
    history: dict[str, list[float]] = {"loss": []}
    if validation_data is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []

    l1 = getattr(config, "l1_penalty", 0.0)
    best_loss = float("inf")
    best_weights = weights.copy()
    best_bias = bias
    stale_epochs = 0
    epochs = max(1, int(config.max_epochs))

    for _ in range(epochs):
        logits = x_train_mapped @ weights + bias
        probabilities = _sigmoid(logits)
        errors = (probabilities - y_train) * sample_weights
        normalizer = max(float(sample_weights.sum()), 1.0)
        gradient_w = (x_train_mapped.T @ errors) / normalizer + l2 * weights
        gradient_b = float(errors.sum() / normalizer)
        weights = weights - learning_rate * gradient_w
        bias = bias - learning_rate * gradient_b

        # Soft-thresholding operator for L1 regularization
        if l1 > 0:
            thresh = learning_rate * l1
            weights = np.sign(weights) * np.maximum(0.0, np.abs(weights) - thresh)

        train_loss = _binary_loss(y_train, _sigmoid(x_train_mapped @ weights + bias), sample_weights, l2, weights)
        if l1 > 0:
            train_loss += l1 * float(np.sum(np.abs(weights)))
        history["loss"].append(train_loss)

        if validation_data is not None:
            x_val, y_val = validation_data
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
            x_val_mapped = feature_map.transform(np.asarray(x_val, dtype=np.float32))
            val_probabilities = _sigmoid(x_val_mapped @ weights + bias)
            val_loss = _binary_loss(y_val, val_probabilities, None, l2, weights)
            if l1 > 0:
                val_loss += l1 * float(np.sum(np.abs(weights)))
            val_accuracy = float(np.mean((val_probabilities >= 0.5).astype(np.int32) == y_val.astype(np.int32)))
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = weights.copy()
                best_bias = bias
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(1, int(config.patience)):
                    break

    if validation_data is not None:
        weights = best_weights
        bias = best_bias
    return (
        NumpyBinaryClassifier(
            weights=weights.astype(np.float32),
            bias=float(bias),
            raw_input_dim=x_train.shape[1],
            feature_map=feature_map.name,
            rff_weights=feature_map.rff_weights,
            rff_bias=feature_map.rff_bias,
        ),
        history,
    )


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
            "TensorFlow is not installed. Install the optional backend with: "
            "python -m pip install -r requirements-tensorflow.txt"
        ) from exc
    return tf


def _try_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return tf


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sample_weights(labels: np.ndarray, class_weight: dict[int, float] | None) -> np.ndarray:
    if not class_weight:
        return np.ones(labels.shape[0], dtype=np.float32)
    return np.asarray([class_weight.get(int(label), 1.0) for label in labels], dtype=np.float32)


def _binary_loss(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray | None,
    l2: float,
    weights: np.ndarray,
) -> float:
    probs = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    losses = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    if sample_weights is not None:
        losses = losses * sample_weights
        denominator = max(float(sample_weights.sum()), 1.0)
    else:
        denominator = max(float(labels.shape[0]), 1.0)
    return float(losses.sum() / denominator + 0.5 * l2 * float(np.sum(weights * weights)))


@dataclass(frozen=True)
class _FeatureMap:
    name: str
    input_dim: int
    rff_weights: np.ndarray | None = None
    rff_bias: np.ndarray | None = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        return _apply_feature_map(values, self.name, self.rff_weights, self.rff_bias)


def _build_feature_map(input_dim: int, config: ModelConfig, rng: np.random.Generator) -> _FeatureMap:
    name = config.feature_map.lower().strip()
    if name not in {"linear", "quadratic", "rff"}:
        raise ValueError("feature_map must be one of: linear, quadratic, rff.")
    if name != "rff":
        return _FeatureMap(name=name, input_dim=input_dim)
    components = max(4, int(config.rff_components))
    gamma = max(float(config.rff_gamma), 1e-6)
    rff_weights = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(input_dim, components)).astype(np.float32)
    rff_bias = rng.uniform(0.0, 2.0 * np.pi, size=components).astype(np.float32)
    return _FeatureMap(name=name, input_dim=input_dim, rff_weights=rff_weights, rff_bias=rff_bias)


def _apply_feature_map(
    values: np.ndarray,
    name: str,
    rff_weights: np.ndarray | None,
    rff_bias: np.ndarray | None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Feature map input must be a 2D array.")
    if name == "linear":
        return array
    if name == "quadratic":
        return _quadratic_features(array)
    if name == "rff":
        if rff_weights is None or rff_bias is None:
            raise ValueError("RFF feature map is missing its random parameters.")
        projection = array @ rff_weights + rff_bias
        return (np.sqrt(2.0 / rff_weights.shape[1]) * np.cos(projection)).astype(np.float32)
    raise ValueError(f"Unsupported feature map: {name}")


def _quadratic_features(values: np.ndarray) -> np.ndarray:
    pieces = [values, values * values]
    interaction_count = 0
    max_interactions = 128
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            pieces.append((values[:, left] * values[:, right]).reshape(-1, 1))
            interaction_count += 1
            if interaction_count >= max_interactions:
                return np.concatenate(pieces, axis=1).astype(np.float32)
    return np.concatenate(pieces, axis=1).astype(np.float32)
