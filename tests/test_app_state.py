from italtensor.app import AppState, _invalidate_model_artifacts, _replace_dataset
from italtensor.data import validate_dataset
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer


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


def test_replace_dataset_invalidates_old_model_state():
    state = AppState(
        model=object(),
        latest_config=ModelConfig(),
        latest_metrics={"f1": 1.0},
        latest_threshold=0.7,
        preprocessor=FeatureStandardizer.identity(1),
    )
    dataset = validate_dataset([[1.0, 2.0], [3.0, 4.0]], [0, 1])

    _replace_dataset(state, dataset)

    assert state.features == [[1.0, 2.0], [3.0, 4.0]]
    assert state.labels == [0, 1]
    assert state.input_dim == 2
    assert state.model is None
    assert state.latest_metrics == {}
