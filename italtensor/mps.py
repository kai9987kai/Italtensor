"""Matrix product state (MPS) binary classifier for ordered tabular features.

Inspired by recent MPS generative / supervised tensor-network work (unitary MPS,
DMRG-style sweeps). Each input dimension is a site; cores are trained with
left-to-right contraction and soft site embeddings for stable gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .modeling import ModelConfig, _sigmoid


def _soft_site_embedding(
    value: float,
    centers: np.ndarray,
    temperature: float = 0.5,
) -> np.ndarray:
    logits = -((float(value) - centers) ** 2) / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-8)
    return weights.astype(np.float32)


@dataclass
class MPSBinaryClassifier:
    """1D MPS chain classifier with bond dimension chi and soft physical bins."""

    cores: list[np.ndarray]
    readout: np.ndarray
    bias: float
    site_centers: np.ndarray
    raw_input_dim: int | None = None
    bond_dim: int = 8
    physical_dim: int = 4
    site_temperature: float = 0.5
    backend: str = "mps-binary"
    calibration_a: float = 1.0
    calibration_b: float = 0.0

    @property
    def input_dim(self) -> int:
        return int(self.raw_input_dim or len(self.cores))

    @property
    def input_shape(self) -> tuple[None, int]:
        return (None, self.input_dim)

    def predict(self, samples: Sequence[float] | np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        sample_array = np.asarray(samples, dtype=np.float32)
        if sample_array.ndim == 1:
            sample_array = sample_array.reshape(1, -1)
        if sample_array.ndim != 2:
            raise ValueError("Prediction input must be one vector or a 2D feature array.")
        if sample_array.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {sample_array.shape[1]}.")
        logits = self._forward_logits(sample_array)
        if self.calibration_a != 1.0 or self.calibration_b != 0.0:
            logits = self.calibration_a * logits + self.calibration_b
        return _sigmoid(logits).reshape(-1, 1).astype(np.float32)

    def _forward_logits(self, features: np.ndarray) -> np.ndarray:
        batch = features.shape[0]
        states = np.ones((batch, 1), dtype=np.float32)
        for site, core in enumerate(self.cores):
            centers = self.site_centers[site]
            phys = np.stack(
                [_soft_site_embedding(features[row, site], centers, self.site_temperature) for row in range(batch)],
                axis=0,
            )
            next_states = np.zeros((batch, core.shape[2]), dtype=np.float32)
            for p in range(core.shape[1]):
                next_states += phys[:, p : p + 1] * (states @ core[:, p, :])
            states = next_states
        logits = states @ self.readout.reshape(-1)
        return (logits.reshape(-1) + self.bias).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_format_version": 1,
            "backend": self.backend,
            "input_dim": self.input_dim,
            "bond_dim": self.bond_dim,
            "physical_dim": self.physical_dim,
            "site_temperature": float(self.site_temperature),
            "cores": [core.astype(float).tolist() for core in self.cores],
            "readout": self.readout.astype(float).tolist(),
            "bias": float(self.bias),
            "site_centers": self.site_centers.astype(float).tolist(),
            "calibration_a": float(self.calibration_a),
            "calibration_b": float(self.calibration_b),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> MPSBinaryClassifier:
        cores_raw = value.get("cores")
        if not isinstance(cores_raw, list) or not cores_raw:
            raise ValueError("MPS model is missing cores.")
        cores = [np.asarray(item, dtype=np.float32) for item in cores_raw]
        readout = np.asarray(value.get("readout"), dtype=np.float32).reshape(-1)
        site_centers = np.asarray(value.get("site_centers"), dtype=np.float32)
        input_dim = int(value.get("input_dim", len(cores)))
        if len(cores) != input_dim:
            raise ValueError("MPS core count does not match input_dim.")
        if site_centers.shape[0] != input_dim:
            raise ValueError("MPS site_centers do not match input_dim.")
        if readout.shape[0] != cores[-1].shape[2]:
            raise ValueError("MPS readout dimension does not match final bond.")
        return cls(
            cores=cores,
            readout=readout,
            bias=float(value.get("bias", 0.0)),
            site_centers=site_centers,
            raw_input_dim=input_dim,
            bond_dim=int(value.get("bond_dim", cores[0].shape[2] if cores else 8)),
            physical_dim=int(value.get("physical_dim", cores[0].shape[1] if cores else 4)),
            site_temperature=float(value.get("site_temperature", 0.5)),
            calibration_a=float(value.get("calibration_a", 1.0)),
            calibration_b=float(value.get("calibration_b", 0.0)),
        )


def _init_cores(
    n_sites: int,
    bond_dim: int,
    physical_dim: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    cores: list[np.ndarray] = []
    chi_left = 1
    for site in range(n_sites):
        chi_right = 1 if site == n_sites - 1 else bond_dim
        core = rng.normal(0.0, 0.05, size=(chi_left, physical_dim, chi_right)).astype(np.float32)
        cores.append(core)
        chi_left = chi_right
    return cores


def _init_site_centers(features: np.ndarray, physical_dim: int) -> np.ndarray:
    n_sites = features.shape[1]
    centers = np.zeros((n_sites, physical_dim), dtype=np.float32)
    for site in range(n_sites):
        col = features[:, site]
        lo = float(np.min(col))
        hi = float(np.max(col))
        if abs(hi - lo) < 1e-8:
            centers[site, :] = lo
        else:
            centers[site, :] = np.linspace(lo, hi, physical_dim, dtype=np.float32)
    return centers


def train_mps_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    *,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    class_weight: dict[int, float] | None = None,
) -> tuple[MPSBinaryClassifier, dict[str, list[float]]]:
    """Train an MPS chain classifier with mini-batch SGD on core tensors."""
    x_train = np.asarray(features, dtype=np.float32)
    y_train = np.asarray(labels, dtype=np.float32).reshape(-1)
    if x_train.ndim != 2 or x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training features must be a 2D array matching label count.")

    bond_dim = max(2, int(getattr(config, "mps_bond_dim", 8)))
    physical_dim = max(2, int(getattr(config, "mps_physical_dim", 4)))
    rng = np.random.default_rng(config.random_seed)
    cores = _init_cores(x_train.shape[1], bond_dim, physical_dim, rng)
    readout = rng.normal(0.0, 0.05, size=cores[-1].shape[2]).astype(np.float32)
    bias = 0.0
    site_centers = _init_site_centers(x_train, physical_dim)

    sample_weights = np.ones(y_train.shape[0], dtype=np.float32)
    if class_weight:
        sample_weights = np.asarray([class_weight.get(int(label), 1.0) for label in y_train], dtype=np.float32)

    base_lr = min(max(config.learning_rate, 1e-4), 0.1)
    batch_size = max(1, int(config.batch_size))
    epochs = max(1, int(config.max_epochs))
    history: dict[str, list[float]] = {"loss": []}
    if validation_data is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []

    best_val = float("inf")
    best_state: tuple[list[np.ndarray], np.ndarray, float] | None = None
    stale = 0

    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        rng.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            wb = sample_weights[batch_idx]
            model = MPSBinaryClassifier(
                cores=[core.copy() for core in cores],
                readout=readout.copy(),
                bias=bias,
                site_centers=site_centers,
                raw_input_dim=x_train.shape[1],
                bond_dim=bond_dim,
                physical_dim=physical_dim,
            )
            logits = model._forward_logits(xb)
            probs = _sigmoid(logits)
            errors = (probs - yb) * wb
            normalizer = max(float(wb.sum()), 1.0)
            grad_logit = errors / normalizer

            grad_readout = np.zeros_like(readout)
            grad_bias = 0.0
            grad_cores = [np.zeros_like(core) for core in cores]

            for row in range(xb.shape[0]):
                g = float(grad_logit[row])
                states: list[np.ndarray] = [np.ones((1,), dtype=np.float32)]
                phys_list: list[np.ndarray] = []
                for site, core in enumerate(cores):
                    phys = _soft_site_embedding(xb[row, site], site_centers[site], model.site_temperature)
                    phys_list.append(phys)
                    nxt = np.zeros(core.shape[2], dtype=np.float32)
                    for p in range(core.shape[1]):
                        nxt += phys[p] * (states[-1] @ core[:, p, :])
                    states.append(nxt)
                final = states[-1]
                grad_readout += g * final
                grad_bias += g
                delta = g * readout
                for site in range(len(cores) - 1, -1, -1):
                    core = cores[site]
                    phys = phys_list[site]
                    left = states[site]
                    for p in range(core.shape[1]):
                        grad_cores[site][:, p, :] += delta.reshape(1, -1) * (left.reshape(-1, 1) * phys[p])
                    if site > 0:
                        back = np.zeros_like(left)
                        for p in range(core.shape[1]):
                            back += phys[p] * (delta.reshape(1, -1) @ core[:, p, :].T).reshape(-1)
                        delta = back

            readout -= base_lr * grad_readout
            bias -= base_lr * grad_bias
            for idx, core in enumerate(cores):
                cores[idx] = core - base_lr * grad_cores[idx]

            probs = _sigmoid(model._forward_logits(xb))
            clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
            batch_loss = float(
                -np.mean(wb * (yb * np.log(clipped) + (1.0 - yb) * np.log(1.0 - clipped))) / max(float(wb.mean()), 1e-8)
            )
            epoch_loss += batch_loss
            n_batches += 1

        history["loss"].append(epoch_loss / max(n_batches, 1))

        if validation_data is not None:
            x_val, y_val = validation_data
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
            val_model = MPSBinaryClassifier(
                cores=cores,
                readout=readout,
                bias=bias,
                site_centers=site_centers,
                raw_input_dim=x_train.shape[1],
                bond_dim=bond_dim,
                physical_dim=physical_dim,
            )
            val_probs = _sigmoid(val_model._forward_logits(np.asarray(x_val, dtype=np.float32)))
            clipped = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
            val_loss = float(-np.mean(y_val * np.log(clipped) + (1.0 - y_val) * np.log(1.0 - clipped)))
            val_acc = float(np.mean((val_probs >= 0.5).astype(np.int32) == y_val.astype(np.int32)))
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if val_loss < best_val:
                best_val = val_loss
                best_state = ([core.copy() for core in cores], readout.copy(), bias)
                stale = 0
            else:
                stale += 1
                if stale >= max(1, int(config.patience)):
                    break

    if best_state is not None:
        cores, readout, bias = best_state

    return (
        MPSBinaryClassifier(
            cores=cores,
            readout=readout,
            bias=float(bias),
            site_centers=site_centers,
            raw_input_dim=x_train.shape[1],
            bond_dim=bond_dim,
            physical_dim=physical_dim,
        ),
        history,
    )
