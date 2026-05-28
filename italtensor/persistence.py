from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import Dataset, dataset_from_jsonable, dataset_to_jsonable
from .experiments import EnsemblePredictor
from .modeling import ModelConfig, NumpyBinaryClassifier
from .mps import MPSBinaryClassifier
from .preprocessing import FeatureStandardizer
from .registry import ModelSlot


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
    trial_history: list[dict[str, Any]] | None = None,
    uncertainty_metadata: dict[str, Any] | None = None,
    ablation_report: dict[str, Any] | None = None,
    decision_curve_report: dict[str, Any] | None = None,
    conformal_set_report: dict[str, Any] | None = None,
    calibration_repair_report: dict[str, Any] | None = None,
    selective_risk_report: dict[str, Any] | None = None,
    sample_review_report: dict[str, Any] | None = None,
    threshold_report: dict[str, Any] | None = None,
    model_response_report: dict[str, Any] | None = None,
    pairwise_interaction_report: dict[str, Any] | None = None,
    slice_report: dict[str, Any] | None = None,
    subgroup_disparity_report: dict[str, Any] | None = None,
    stress_report: dict[str, Any] | None = None,
    permutation_null_report: dict[str, Any] | None = None,
    population_drift_report: dict[str, Any] | None = None,
    adversarial_validation_report: dict[str, Any] | None = None,
    chronological_holdout_report: dict[str, Any] | None = None,
    cartography_report: dict[str, Any] | None = None,
    ood_sentinel_report: dict[str, Any] | None = None,
    bootstrap_stability_report: dict[str, Any] | None = None,
    prototype_audit_report: dict[str, Any] | None = None,
    feature_separability_report: dict[str, Any] | None = None,
    neighborhood_hardness_report: dict[str, Any] | None = None,
    dataset_triage_report: dict[str, Any] | None = None,
    experiment_advisor_report: dict[str, Any] | None = None,
    trial_inspector_report: dict[str, Any] | None = None,
    promotion_gate_report: dict[str, Any] | None = None,
    mps_sweep_report: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    model_path = Path(path)
    is_ensemble = isinstance(model, EnsemblePredictor)
    is_numpy_model = isinstance(model, NumpyBinaryClassifier)
    is_mps_model = isinstance(model, MPSBinaryClassifier)
    if is_ensemble:
        model_path = model_path.with_suffix(".italtensor-ensemble.json")
    elif is_mps_model:
        model_path = model_path.with_suffix(".italtensor-mps.json")
    elif is_numpy_model:
        model_path = model_path.with_suffix(".italtensor-model.json")
    elif model_path.suffix != ".keras":
        model_path = model_path.with_suffix(".keras")

    resolved_preprocessor = preprocessor or FeatureStandardizer.identity(input_dim)
    if resolved_preprocessor.selected_indices is None:
        if resolved_preprocessor.mean.shape[0] != input_dim:
            raise ValueError(
                f"Preprocessing metadata expects {resolved_preprocessor.mean.shape[0]} features, "
                f"model expects {input_dim}."
            )
    else:
        max_idx = max(resolved_preprocessor.selected_indices) if resolved_preprocessor.selected_indices else 0
        if max_idx >= input_dim:
            raise ValueError(
                f"Feature selection index {max_idx} is out of bounds for input dimension {input_dim}."
            )

    if is_ensemble or is_numpy_model or is_mps_model:
        model_path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    else:
        model.save(str(model_path))
    metadata_path = model_metadata_path(model_path)
    metadata = {
        "model_format_version": 1,
        "model_backend": (
            "ensemble"
            if is_ensemble
            else (
                "mps-binary"
                if is_mps_model
                else ("numpy-logistic" if is_numpy_model else "tensorflow-keras")
            )
        ),
        "model_feature_map": getattr(model, "feature_map", None),
        "input_dim": input_dim,
        "label_schema": {"negative": 0, "positive": 1},
        "best_config": config.to_dict(),
        "validation_metrics": metrics or {},
        "uncertainty": uncertainty_metadata or {},
        "feature_ablation_diagnostics": ablation_report or None,
        "decision_curve_diagnostics": decision_curve_report or None,
        "posthoc_conformal_diagnostics": conformal_set_report or None,
        "posthoc_calibration_repair_diagnostics": calibration_repair_report or None,
        "selective_prediction_diagnostics": selective_risk_report or None,
        "sample_review": sample_review_report or None,
        "threshold_diagnostics": threshold_report or None,
        "model_response_diagnostics": model_response_report or None,
        "pairwise_interaction_diagnostics": pairwise_interaction_report or None,
        "slice_diagnostics": slice_report or None,
        "subgroup_disparity_diagnostics": subgroup_disparity_report or None,
        "stress_lab": stress_report or None,
        "posthoc_permutation_null_diagnostics": permutation_null_report or None,
        "population_drift_diagnostics": population_drift_report or None,
        "adversarial_validation_diagnostics": adversarial_validation_report or None,
        "chronological_holdout_diagnostics": chronological_holdout_report or None,
        "dataset_cartography": cartography_report or None,
        "ood_sentinel": ood_sentinel_report or None,
        "bootstrap_stability_diagnostics": bootstrap_stability_report or None,
        "prototype_audit": prototype_audit_report or None,
        "feature_separability": feature_separability_report or None,
        "neighborhood_hardness": neighborhood_hardness_report or None,
        "dataset_triage": dataset_triage_report or None,
        "experiment_advisor": experiment_advisor_report or None,
        "trial_inspector": trial_inspector_report or None,
        "promotion_gate": promotion_gate_report or None,
        "mps_bond_sweep": mps_sweep_report or None,
        "threshold": float(threshold),
        "preprocessing": resolved_preprocessor.to_dict(),
        "feature_importances": feature_importances or [],
        "trial_history": trial_history or [],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_model_bundle(path: str | Path):
    model_path = Path(path)
    if model_path.suffix == ".json":
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if payload.get("ensemble_format_version") is not None:
            model = EnsemblePredictor.from_dict(payload)
        elif payload.get("backend") == "mps-binary":
            model = MPSBinaryClassifier.from_dict(payload)
        elif payload.get("backend") == "numpy-logistic":
            model = NumpyBinaryClassifier.from_dict(payload)
        else:
            raise ValueError("Unsupported JSON model file.")
    else:
        try:
            tf = _tensorflow()
        except RuntimeError as exc:
            raise RuntimeError(
                ".keras models require the optional TensorFlow backend. "
                "Install it with: python -m pip install -r requirements-tensorflow.txt. "
                "Without TensorFlow, load an .italtensor-model.json fallback model instead."
            ) from exc
        model = tf.keras.models.load_model(str(model_path))

    metadata_path = model_metadata_path(model_path)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def model_metadata_path(model_path: str | Path) -> Path:
    """Sidecar metadata path: model.italtensor-model.json -> model.italtensor-meta.json."""
    path = Path(model_path)
    name = path.name
    for suffix in (
        ".italtensor-ensemble.json",
        ".italtensor-mps.json",
        ".italtensor-model.json",
        ".keras",
    ):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            return path.with_name(f"{stem}.italtensor-meta.json")
    return path.with_suffix(path.suffix + ".italtensor-meta.json")


def save_model_registry(path: str | Path, slots: list[ModelSlot], *, input_dim: int | None = None) -> Path:
    """Persist in-memory model slots (NumPy models and ensembles)."""
    serialized: list[dict[str, Any]] = []
    for slot in slots:
        model = slot.model
        if isinstance(model, EnsemblePredictor):
            model_payload = {"kind": "ensemble", "data": model.to_dict()}
        elif isinstance(model, NumpyBinaryClassifier):
            model_payload = {"kind": "numpy", "data": model.to_dict()}
        elif isinstance(model, MPSBinaryClassifier):
            model_payload = {"kind": "mps", "data": model.to_dict()}
        else:
            raise ValueError(
                f"Slot '{slot.name}' uses a backend that cannot be stored in a registry file. "
                "Save Keras models individually, then reload them into slots."
            )
        serialized.append(
            {
                "name": slot.name,
                "metrics": slot.metrics,
                "threshold": slot.threshold,
                "config": slot.config.to_dict() if slot.config is not None else None,
                "preprocessing": slot.preprocessor.to_dict() if slot.preprocessor is not None else None,
                "model": model_payload,
            }
        )
    output = {
        "kind": "italtensor.model_registry",
        "schema_version": 1,
        "input_dim": input_dim,
        "slots": serialized,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    output_path = Path(path)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def load_model_registry(path: str | Path) -> tuple[list["ModelSlot"], int | None]:
    """Load model slots from a registry JSON file."""
    from .registry import ModelSlot

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "italtensor.model_registry":
        raise ValueError("Not an Italtensor model registry file.")
    slots: list[ModelSlot] = []
    for item in payload.get("slots", []):
        model_info = item.get("model", {})
        kind = model_info.get("kind")
        data = model_info.get("data")
        if kind == "ensemble":
            model = EnsemblePredictor.from_dict(data)
        elif kind == "numpy":
            model = NumpyBinaryClassifier.from_dict(data)
        elif kind == "mps":
            model = MPSBinaryClassifier.from_dict(data)
        else:
            raise ValueError(f"Unsupported registry model kind: {kind}")
        config_raw = item.get("config")
        config = ModelConfig.from_dict(config_raw) if isinstance(config_raw, dict) else ModelConfig()
        preproc_raw = item.get("preprocessing")
        preprocessor = (
            FeatureStandardizer.from_dict(preproc_raw) if isinstance(preproc_raw, dict) else None
        )
        slots.append(
            ModelSlot(
                model=model,
                config=config,
                metrics=item.get("metrics") if isinstance(item.get("metrics"), dict) else {},
                preprocessor=preprocessor,
                threshold=float(item.get("threshold", 0.5)),
                name=str(item.get("name", "Loaded slot")),
            )
        )
    input_dim = payload.get("input_dim")
    return slots, int(input_dim) if input_dim is not None else None


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install the optional backend with: "
            "python -m pip install -r requirements-tensorflow.txt"
        ) from exc
    return tf
