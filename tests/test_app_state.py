import numpy as np
from unittest.mock import patch
from italtensor.app import (
    AppState,
    ModelSlot,
    _apply_preset_metadata,
    _format_uncertainty,
    _invalidate_model_artifacts,
    _replace_dataset,
    _store_model_slot,
    _activate_model_slot,
    _build_ensemble,
    _compare_models,
    _run_weight_analysis,
)
from italtensor.data import validate_dataset
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer


class FakeElement:
    def __init__(self, values=None):
        self.value = ""
        self.values = values or []

    def update(self, value=None, values=None, append=False, **kwargs):
        if value is not None:
            if append:
                self.value = str(self.value) + str(value)
            else:
                self.value = value
        if values is not None:
            self.values = values

    def get_list_values(self):
        return self.values


class FakeWindow(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AllKeysDict = {}

    def __missing__(self, key):
        self.AllKeysDict[key] = True
        self[key] = FakeElement()
        return self[key]


def test_invalidate_model_artifacts_keeps_dataset_shape_but_clears_model_state():
    state = AppState(
        features=[[1.0]],
        labels=[1],
        input_dim=1,
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 1.0},
        latest_threshold=0.7,
        preprocessor=FeatureStandardizer.identity(1),
        feature_importances=[{"feature_index": 0, "importance": 0.5}],
        trial_history=[{"metrics": {"f1": 1.0}}],
        uncertainty_metadata={"conformal_quantile": 0.3},
    )

    _invalidate_model_artifacts(state)

    assert state.features == [[1.0]]
    assert state.labels == [1]
    assert state.input_dim == 1
    assert state.model is None
    assert state.latest_config is None
    assert state.latest_metrics == {}
    assert state.latest_threshold == 0.5
    assert state.preprocessor is None
    assert state.feature_importances == []
    assert state.trial_history == []
    assert state.uncertainty_metadata == {}


def test_replace_dataset_invalidates_old_model_state():
    state = AppState(
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 1.0},
        latest_threshold=0.7,
        preprocessor=FeatureStandardizer.identity(1),
        uncertainty_metadata={"conformal_quantile": 0.3},
    )
    dataset = validate_dataset([[1.0, 2.0], [3.0, 4.0]], [0, 1])

    _replace_dataset(state, dataset)

    assert state.features == [[1.0, 2.0], [3.0, 4.0]]
    assert state.labels == [0, 1]
    assert state.input_dim == 2
    assert state.model is None
    assert state.latest_metrics == {}
    assert state.uncertainty_metadata == {}


def test_format_uncertainty_includes_source_and_coverage():
    summary = _format_uncertainty(
        {
            "conformal_source": "dedicated_calibration",
            "conformal_alpha": 0.1,
            "conformal_target_coverage": 0.9,
            "conformal_coverage": 0.85,
        }
    )

    assert "source=dedicated_calibration" in summary
    assert "conformal_target_coverage=0.9000" in summary
    assert "conformal_coverage=0.8500" in summary


def test_apply_preset_metadata_updates_sparse_training_defaults():
    window = FakeWindow()

    _apply_preset_metadata(
        window,
        {
            "recommended_feature_map": "quadratic",
            "training_defaults": {
                "epochs": 90,
                "batch_size": 16,
                "trials": 16,
                "feature_map": "quadratic",
                "l1_penalty": 0.001,
                "feature_selection_k": 6,
            },
        },
    )

    assert window["-FEATURE_MAP-"].value == "quadratic"
    assert window["-L1_PENALTY-"].value == "0.001"
    assert window["-FEATURE_K-"].value == "6"


def test_store_model_slot():
    window = FakeWindow()
    state = AppState()
    state.model = object()
    state.latest_config = ModelConfig()
    state.latest_metrics = {"f1": 0.8}
    state.latest_threshold = 0.5

    with patch("PySimpleGUI.popup_get_text", return_value="Model A"):
        _store_model_slot(window, state, {})

    assert len(state.model_slots) == 1
    assert state.model_slots[0].name == "Model A"
    assert state.model_slots[0].metrics["f1"] == 0.8
    assert state.active_slot_index == 0


def test_activate_model_slot():
    window = FakeWindow()
    state = AppState()
    slot = ModelSlot(
        model=object(),
        config=ModelConfig(learning_rate=0.05),
        metrics={"f1": 0.9},
        preprocessor=None,
        threshold=0.6,
        name="Model B",
    )
    state.model_slots.append(slot)
    state.active_slot_index = 0

    window["-MODEL_SLOTS-"].values = ["* Model B (F1: 0.9000)"]

    _activate_model_slot(window, state, {"-MODEL_SLOTS-": ["* Model B (F1: 0.9000)"]})

    assert state.active_slot_index == 0
    assert state.latest_config.learning_rate == 0.05
    assert state.latest_threshold == 0.6


def test_build_ensemble():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier
    from italtensor.preprocessing import FeatureStandardizer

    model_a = NumpyBinaryClassifier(weights=np.array([1.0]), bias=0.0, raw_input_dim=1)
    slot = ModelSlot(
        model=model_a,
        config=ModelConfig(),
        metrics={"f1": 0.8},
        preprocessor=FeatureStandardizer.identity(1),
        threshold=0.5,
        name="Model A",
    )
    state.model_slots.append(slot)
    state.features = [[1.0], [2.0], [3.0], [4.0]]
    state.labels = [0, 1, 0, 1]

    _build_ensemble(window, state, {})

    assert len(state.model_slots) == 2
    assert "Ensemble" in state.model_slots[1].name


def test_compare_models():
    window = FakeWindow()
    state = AppState()
    slot = ModelSlot(
        model=object(),
        config=ModelConfig(),
        metrics={"f1": 0.8, "accuracy": 0.85, "brier_score": 0.1, "ece": 0.05},
        preprocessor=None,
        threshold=0.5,
        name="Model A",
    )
    state.model_slots.append(slot)

    _compare_models(window, state, {})
    assert "Model A" in window["-LOG-"].value


def test_run_weight_analysis():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier

    state.model = NumpyBinaryClassifier(weights=np.array([1.0, 0.0]), bias=0.5, raw_input_dim=2)

    _run_weight_analysis(window, state, {})
    assert "Weight Analysis" in window["-LOG-"].value
    assert "Sparsity" in window["-LOG-"].value
