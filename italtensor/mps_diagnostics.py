"""MPS expressivity diagnostics: bond and site-order sweeps.

Higher bond dimension chi increases representational capacity; sweeps help find a
practical chi before spending epochs on a single setting. Site order also matters
for MPS/TT-style models because each feature is embedded at a chain position.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from .data import Dataset
from .experiments import evaluate_predictions, split_train_validation
from .modeling import ModelConfig
from .mps import train_mps_model
from .preprocessing import FeatureStandardizer

OrderSpec = Mapping[str, Sequence[int]] | Sequence[tuple[str, Sequence[int]]]


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


def run_mps_order_sweep(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    config: ModelConfig,
    *,
    orders: OrderSpec | None = None,
    validation_fraction: float = 0.25,
    seed: int = 42,
    max_orders: int = 5,
) -> dict[str, Any]:
    """Retrain small MPS models over candidate feature-site orders.

    The active model is not mutated. Each candidate order trains a fresh MPS on
    standardized, permuted feature columns, then ranks validation results by F1,
    accuracy, and Brier score.
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("MPS order sweep requires aligned 2D features and labels.")
    if x.shape[0] < 8:
        raise ValueError("MPS order sweep needs at least 8 samples.")
    if np.unique(y).size < 2:
        raise ValueError("MPS order sweep needs both classes present.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("MPS order sweep validation_fraction must be between 0 and 1.")

    input_dim = int(x.shape[1])
    if input_dim < 2:
        raise ValueError("MPS order sweep needs at least 2 features.")
    capped_orders = max(1, int(max_orders))
    candidate_orders = _resolve_candidate_orders(x, y, orders=orders, seed=seed, max_orders=capped_orders)

    train_ratio = 1.0 - float(validation_fraction)
    x_train, y_train, x_val, y_val = split_train_validation(
        Dataset(features=x, labels=y, input_dim=input_dim),
        train_ratio=train_ratio,
        seed=seed,
    )
    preprocessor = FeatureStandardizer.fit(x_train)
    x_train_std = preprocessor.transform(x_train)
    x_val_std = preprocessor.transform(x_val)

    rows: list[dict[str, Any]] = []
    for order_name, order in candidate_orders:
        order_array = np.asarray(order, dtype=np.int32)
        trial_config = ModelConfig.from_dict({**config.to_dict(), "backend": "mps"})
        model, history = train_mps_model(
            x_train_std[:, order_array],
            y_train,
            trial_config,
            validation_data=(x_val_std[:, order_array], y_val),
        )
        val_probs = model.predict(x_val_std[:, order_array]).reshape(-1)
        metrics = evaluate_predictions(y_val, val_probs, threshold=0.5)
        validation_loss = _last_history_value(history, "val_loss")
        row = {
            "order_name": order_name,
            "order": [int(index) for index in order],
            "feature_order_1_based": [int(index) + 1 for index in order],
            "is_original": tuple(order) == tuple(range(input_dim)),
            "f1": float(metrics.get("f1", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "brier_score": float(metrics.get("brier_score", 0.0)),
            "ece": float(metrics.get("ece", 0.0)),
            "validation_loss": float(validation_loss) if validation_loss is not None else None,
            "final_train_loss": _last_history_value(history, "loss") or 0.0,
            "epochs_run": len(history.get("loss", [])),
        }
        rows.append(row)

    rows.sort(
        key=lambda item: (
            float(item.get("f1", 0.0)),
            float(item.get("accuracy", 0.0)),
            -float(item.get("brier_score", 0.0)),
            1 if item.get("is_original") else 0,
        ),
        reverse=True,
    )
    best = rows[0]
    original = next((row for row in rows if row.get("is_original")), None)
    original_f1 = float(original.get("f1", 0.0)) if original else None
    best_delta = float(best.get("f1", 0.0)) - original_f1 if original_f1 is not None else 0.0
    material_gain = best_delta >= 0.02 and not best.get("is_original", False)
    return {
        "input_dim": input_dim,
        "physical_dim": int(getattr(config, "mps_physical_dim", 4)),
        "bond_dim": int(getattr(config, "mps_bond_dim", 8)),
        "validation_samples": int(y_val.shape[0]),
        "orders_tested": [row["order_name"] for row in rows],
        "results": rows,
        "recommended_order_name": str(best["order_name"]),
        "recommended_order": [int(index) for index in best["order"]],
        "recommended_feature_order_1_based": [int(index) for index in best["feature_order_1_based"]],
        "recommended_f1": float(best.get("f1", 0.0)),
        "original_f1": original_f1,
        "best_delta_f1_vs_original": float(best_delta),
        "material_gain": bool(material_gain),
        "adoption_note": (
            "This is site-order sensitivity evidence, not feature importance. "
            "To adopt a non-original order, retrain and save a model with CSV columns or manual vectors "
            "consistently reordered to match recommended_feature_order_1_based."
        ),
    }


def format_mps_order_sweep_summary(report: dict[str, Any]) -> str:
    name = str(report.get("recommended_order_name", "-"))
    f1 = float(report.get("recommended_f1", 0.0))
    delta = float(report.get("best_delta_f1_vs_original", 0.0))
    gain = "material gain" if report.get("material_gain") else "no material gain"
    return f"MPS order sweep: recommended order={name} (val F1={f1:.4f}, delta vs original={delta:+.4f}, {gain})"


def _resolve_candidate_orders(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    orders: OrderSpec | None,
    seed: int,
    max_orders: int,
) -> list[tuple[str, tuple[int, ...]]]:
    input_dim = int(features.shape[1])
    if orders is None:
        candidates = _default_candidate_orders(features, labels, seed=seed)
    elif isinstance(orders, Mapping):
        candidates = [(str(name), tuple(order)) for name, order in orders.items()]
    else:
        candidates = [(str(name), tuple(order)) for name, order in orders]

    resolved: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, raw_order in candidates:
        order = tuple(int(index) for index in raw_order)
        _validate_order(order, input_dim)
        if order in seen:
            continue
        resolved.append((name or f"order_{len(resolved) + 1}", order))
        seen.add(order)
        if len(resolved) >= max_orders:
            break
    if not resolved:
        raise ValueError("MPS order sweep needs at least one valid feature order.")
    if tuple(range(input_dim)) not in seen:
        original = tuple(range(input_dim))
        resolved = [("original", original)] + resolved[: max_orders - 1]
    return resolved


def _default_candidate_orders(features: np.ndarray, labels: np.ndarray, *, seed: int) -> list[tuple[str, tuple[int, ...]]]:
    input_dim = int(features.shape[1])
    original = tuple(range(input_dim))
    rng = np.random.default_rng(seed)
    return [
        ("original", original),
        ("reversed", tuple(reversed(original))),
        ("label_correlation", _label_correlation_order(features, labels)),
        ("correlation_path", _correlation_path_order(features, labels)),
        ("seeded_random", tuple(int(index) for index in rng.permutation(input_dim))),
    ]


def _validate_order(order: tuple[int, ...], input_dim: int) -> None:
    if len(order) != input_dim or set(order) != set(range(input_dim)):
        raise ValueError(f"MPS order sweep feature order must be a permutation of 0..{input_dim - 1}.")


def _label_correlation_order(features: np.ndarray, labels: np.ndarray) -> tuple[int, ...]:
    scores = _label_correlation_scores(features, labels)
    return tuple(int(index) for index in np.argsort(-scores, kind="stable"))


def _correlation_path_order(features: np.ndarray, labels: np.ndarray) -> tuple[int, ...]:
    input_dim = int(features.shape[1])
    label_scores = _label_correlation_scores(features, labels)
    feature_corr = _safe_abs_corrcoef(features)
    start = int(np.argmax(label_scores))
    order = [start]
    unused = set(range(input_dim))
    unused.remove(start)
    while unused:
        last = order[-1]
        next_index = max(
            unused,
            key=lambda index: (
                float(feature_corr[last, index]),
                float(label_scores[index]),
                -int(index),
            ),
        )
        order.append(int(next_index))
        unused.remove(next_index)
    return tuple(order)


def _label_correlation_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centered_y = labels.astype(np.float64) - float(np.mean(labels))
    y_std = float(np.std(centered_y))
    scores = np.zeros(features.shape[1], dtype=np.float64)
    if y_std <= 1e-12:
        return scores
    for index in range(features.shape[1]):
        column = features[:, index].astype(np.float64)
        column = column - float(np.mean(column))
        x_std = float(np.std(column))
        if x_std <= 1e-12:
            continue
        corr = float(np.mean(column * centered_y) / (x_std * y_std))
        scores[index] = abs(corr) if np.isfinite(corr) else 0.0
    return scores


def _safe_abs_corrcoef(features: np.ndarray) -> np.ndarray:
    input_dim = int(features.shape[1])
    corr = np.eye(input_dim, dtype=np.float64)
    for left in range(input_dim):
        left_col = features[:, left].astype(np.float64)
        left_col = left_col - float(np.mean(left_col))
        left_std = float(np.std(left_col))
        if left_std <= 1e-12:
            continue
        for right in range(left + 1, input_dim):
            right_col = features[:, right].astype(np.float64)
            right_col = right_col - float(np.mean(right_col))
            right_std = float(np.std(right_col))
            if right_std <= 1e-12:
                continue
            value = float(np.mean(left_col * right_col) / (left_std * right_std))
            corr[left, right] = corr[right, left] = abs(value) if np.isfinite(value) else 0.0
    return corr


def _last_history_value(history: dict[str, Sequence[float]], key: str) -> float | None:
    values = history.get(key) or []
    if not values:
        return None
    return float(values[-1])
