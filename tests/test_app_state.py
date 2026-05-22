from italtensor.app import AppState, _apply_preset_metadata, _format_uncertainty, _invalidate_model_artifacts, _replace_dataset
from italtensor.data import validate_dataset
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer


class FakeElement:
    def __init__(self):
        self.value = None

    def update(self, value=None, **kwargs):
        self.value = value


class FakeWindow(dict):
    def __missing__(self, key):
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
