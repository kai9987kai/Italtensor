import numpy as np
import pytest
from unittest.mock import patch
from italtensor.app import (
    AppState,
    _apply_preset_metadata,
    _format_uncertainty,
    _invalidate_model_artifacts,
    _replace_dataset,
    _store_model_slot,
    _activate_model_slot,
    _build_ensemble,
    _compare_models,
    _run_weight_analysis,
    _handle_worker_done,
    _import_reviewed_labels,
    _run_shap_analysis,
    _run_decision_boundary,
)
from italtensor.data import DataValidationError, validate_dataset
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer
from italtensor.registry import ModelSlot


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
        latest_ablation_report={"summary": {"top_feature": "x1"}},
        latest_sample_review_report={"summary": {"label_issue_count": 1}},
        latest_threshold_report={"summary": {"best_f1": 1.0}},
        latest_slice_report={"summary": {"worst_f1_delta": -0.5}},
        latest_stress_report={"summary": {"worst_f1": 0.5}},
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
    assert state.latest_ablation_report is None
    assert state.latest_sample_review_report is None
    assert state.latest_threshold_report is None
    assert state.latest_slice_report is None
    assert state.latest_stress_report is None


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


def test_handle_worker_done_stores_stress_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "base": {"f1": 0.9},
        "summary": {
            "worst_f1": 0.7,
            "stress_f1_ratio": 0.7777,
            "max_label_flip_rate": 0.25,
            "worst_case": "feature_dropout@0.25",
        },
        "perturbations": [
            {
                "kind": "feature_dropout",
                "level": 0.25,
                "f1": 0.7,
                "label_flip_rate": 0.25,
                "mean_probability_shift": 0.12,
            }
        ],
    }

    _handle_worker_done(window, state, ("stress_test", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_stress_report == report
    assert state.busy is False
    assert "Stress suite" in window["-LOG-"].value


def test_handle_worker_done_stores_ablation_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "base": {"f1": 0.9},
        "summary": {
            "top_feature": "x1",
            "max_f1_drop": 0.4,
            "max_label_flip_rate": 0.25,
            "label_proxy_count": 1,
        },
        "features": [
            {
                "feature_index": 0,
                "f1_drop": 0.4,
                "permutation_f1_drop": 0.3,
                "label_flip_rate": 0.25,
                "permutation_label_flip_rate": 0.1,
                "label_correlation": 0.9,
                "risk_flags": ["label_proxy"],
            }
        ],
    }

    _handle_worker_done(window, state, ("ablation_diagnostics", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_ablation_report == report
    assert state.busy is False
    assert "Ablation diagnostics" in window["-LOG-"].value


def test_handle_worker_done_stores_slice_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "base": {"f1": 0.9},
        "summary": {
            "slice_count": 1,
            "worst_slice": "x2[0, 1]",
            "worst_f1_delta": -0.4,
            "worst_accuracy_delta": -0.25,
        },
        "slices": [
            {
                "feature_index": 1,
                "left": 0.0,
                "right": 1.0,
                "count": 4,
                "f1": 0.5,
                "accuracy": 0.5,
                "f1_delta": -0.4,
            }
        ],
    }

    _handle_worker_done(window, state, ("slice_diagnostics", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_slice_report == report
    assert state.busy is False
    assert "Slice diagnostics" in window["-LOG-"].value


def test_handle_worker_done_stores_threshold_report_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "current_threshold": 0.4,
        "summary": {
            "best_f1_threshold": 0.3,
            "min_cost_threshold": 0.2,
            "best_f1": 0.95,
            "min_cost": 0.1,
            "current_cost": 0.2,
        },
        "best_f1": {"threshold": 0.3, "f1": 0.95, "precision": 0.9, "recall": 1.0, "cost": 0.15},
        "best_balanced_accuracy": {"threshold": 0.35, "f1": 0.9, "precision": 0.9, "recall": 0.9, "cost": 0.2},
        "min_cost": {"threshold": 0.2, "f1": 0.85, "precision": 0.8, "recall": 1.0, "cost": 0.1},
    }

    _handle_worker_done(window, state, ("threshold_diagnostics", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_threshold_report == report
    assert state.latest_threshold == 0.4
    assert state.busy is False
    assert "Threshold sweep" in window["-LOG-"].value


def test_handle_worker_done_stores_sample_review_without_mutating_model():
    window = FakeWindow()
    state = AppState(model=object(), latest_metrics={"f1": 0.9}, latest_threshold=0.4, busy=True)
    model = state.model
    report = {
        "summary": {
            "label_issue_count": 1,
            "disagreement_count": 2,
            "ambiguous_count": 1,
            "mean_loss": 0.3,
            "max_loss": 1.2,
        },
        "label_issues": [
            {"row_index": 2, "label": 0, "predicted_label": 1, "probability": 0.95, "loss": 2.9}
        ],
        "hard_examples": [],
        "ambiguous_examples": [],
    }

    _handle_worker_done(window, state, ("sample_review", report))

    assert state.model is model
    assert state.latest_metrics == {"f1": 0.9}
    assert state.latest_sample_review_report == report
    assert state.latest_threshold == 0.4
    assert state.busy is False
    assert "Sample review" in window["-LOG-"].value


def test_import_reviewed_labels_appends_rows_and_invalidates_model(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text(
        "\n".join(
            [
                "x1,x2,italtensor_probability,italtensor_label,italtensor_review_label",
                "0.2,0.3,0.4,0,1",
                "0.4,0.5,0.6,1,",
                "0.6,0.7,0.8,1,0",
            ]
        ),
        encoding="utf-8",
    )
    window = FakeWindow()
    state = AppState(
        features=[[0.0, 0.0], [1.0, 1.0]],
        labels=[0, 1],
        input_dim=2,
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 0.8},
    )

    _import_reviewed_labels(window, state, {"-BATCH_OUTPUT_PATH-": str(path)})

    assert len(state.labels) == 4
    assert state.labels[-2:] == [1, 0]
    np.testing.assert_allclose(state.features[-2:], [[0.2, 0.3], [0.6, 0.7]])
    assert state.model is None
    assert state.latest_metrics == {}
    assert "Imported 2 reviewed label" in window["-LOG-"].value


def test_import_reviewed_labels_with_no_reviewed_rows_does_not_mutate(tmp_path):
    path = tmp_path / "reviewed.csv"
    path.write_text("x1,italtensor_probability,italtensor_review_label\n0.2,0.4,\n", encoding="utf-8")
    window = FakeWindow()
    state = AppState(features=[[0.0]], labels=[0], input_dim=1)

    with pytest.raises(DataValidationError):
        _import_reviewed_labels(window, state, {"-BATCH_OUTPUT_PATH-": str(path)})

    assert state.features == [[0.0]]
    assert state.labels == [0]


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


def test_run_shap_analysis():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier
    from italtensor.preprocessing import FeatureStandardizer

    # 1. Test fallback when model is None
    _run_shap_analysis(window, state, {})
    assert "Train or load a model first" in window["-LOG-"].value

    # 2. Test when prediction vector is missing
    state.model = NumpyBinaryClassifier(weights=np.array([1.0, 0.0]), bias=0.5, raw_input_dim=2)
    window["-LOG-"].value = ""
    _run_shap_analysis(window, state, {})
    assert "Please enter a prediction vector JSON" in window["-LOG-"].value

    # 3. Test successful SHAP run
    window["-LOG-"].value = ""
    state.preprocessor = FeatureStandardizer(mean=np.array([0.0, 0.0]), scale=np.array([1.0, 1.0]), selected_indices=[0, 1])
    values = {"-PREDICTION_VECTOR-": "[1.0, -1.0]"}
    _run_shap_analysis(window, state, values)
    assert "SHAP Local Feature Attributions" in window["-LOG-"].value
    assert "x1" in window["-LOG-"].value


def test_run_decision_boundary():
    window = FakeWindow()
    state = AppState()
    from italtensor.modeling import NumpyBinaryClassifier
    from italtensor.preprocessing import FeatureStandardizer

    # 1. Test fallback when model is None
    _run_decision_boundary(window, state, {})
    assert "Train or load a model first" in window["-LOG-"].value

    # 2. Test when dataset is missing
    state.model = NumpyBinaryClassifier(weights=np.array([1.0, 0.0]), bias=0.5, raw_input_dim=2)
    window["-LOG-"].value = ""
    _run_decision_boundary(window, state, {})
    assert "No dataset loaded" in window["-LOG-"].value

    # 3. Test successful decision boundary run
    window["-LOG-"].value = ""
    state.features = [[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]]
    state.labels = [0, 1, 0, 1]
    state.preprocessor = FeatureStandardizer(mean=np.array([0.0, 0.0]), scale=np.array([1.0, 1.0]), selected_indices=[0, 1])
    
    _run_decision_boundary(window, state, {})
    assert "Decision Boundary Visualization" in window["-LOG-"].value
