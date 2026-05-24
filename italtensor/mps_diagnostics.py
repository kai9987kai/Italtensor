"""MPS expressivity diagnostics: bond-dimension sweeps (TeNPy / ITensor-style practice).

Higher bond dimension chi increases representational capacity; sweeps help find a
practical chi before spending epochs on a single setting.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .data import Dataset
from .experiments import evaluate_predictions, split_train_validation
from .modeling import ModelConfig
from .mps import train_mps_model
from .preprocessing import FeatureStandardizer


def run_mps_bond_sweep(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    config: ModelConfig,
    *,
    bond_dims: Sequence[int] | None = None,
    validation_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Train MPS models at several bond dimensions and rank validation F1."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("MPS bond sweep requires aligned 2D features and labels.")
    if x.shape[0] < 8:
        raise ValueError("MPS bond sweep needs at least 8 samples.")
    if np.unique(y).size < 2:
        raise ValueError("MPS bond sweep needs both classes present.")

    dims = sorted({max(2, int(chi)) for chi in (bond_dims or (4, 8, 16, 24))})
    train_ratio = 1.0 - float(validation_fraction)
    x_train, y_train, x_val, y_val = split_train_validation(
        Dataset(features=x, labels=y, input_dim=x.shape[1]),
        train_ratio=train_ratio,
        seed=seed,
    )
    preprocessor = FeatureStandardizer.fit(x_train)
    x_train_std = preprocessor.transform(x_train)
    x_val_std = preprocessor.transform(x_val)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for chi in dims:
        trial_config = ModelConfig.from_dict({**config.to_dict(), "mps_bond_dim": chi, "backend": "mps"})
        model, history = train_mps_model(
            x_train_std,
            y_train,
            trial_config,
            validation_data=(x_val_std, y_val),
        )
        val_probs = model.predict(x_val_std).reshape(-1)
        metrics = evaluate_predictions(y_val, val_probs, threshold=0.5)
        row = {
            "bond_dim": chi,
            "f1": float(metrics.get("f1", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "brier_score": float(metrics.get("brier_score", 0.0)),
            "ece": float(metrics.get("ece", 0.0)),
            "final_train_loss": float(history["loss"][-1]) if history.get("loss") else 0.0,
            "epochs_run": len(history.get("loss", [])),
        }
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best = row

    rows.sort(key=lambda item: item["f1"], reverse=True)
    return {
        "input_dim": int(x.shape[1]),
        "physical_dim": int(getattr(config, "mps_physical_dim", 4)),
        "validation_samples": int(y_val.shape[0]),
        "bond_dims_tested": dims,
        "results": rows,
        "recommended_bond_dim": int(best["bond_dim"]) if best else dims[0],
        "recommended_f1": float(best["f1"]) if best else 0.0,
    }


def format_mps_sweep_summary(report: dict[str, Any]) -> str:
    rec = int(report.get("recommended_bond_dim", 0))
    f1 = float(report.get("recommended_f1", 0.0))
    tested = report.get("bond_dims_tested", [])
    return f"MPS bond sweep: tested chi={list(tested)}, recommended chi={rec} (val F1={f1:.4f})"
