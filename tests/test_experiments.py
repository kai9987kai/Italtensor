import pytest
import numpy as np

from italtensor.experiments import (
    ExperimentResult,
    balanced_class_weights,
    evaluate_predictions,
    generate_random_configs,
    optimize_threshold,
    permutation_feature_importance,
    run_experiments,
    select_best_result,
    split_train_validation,
)
from italtensor.data import validate_dataset
from italtensor.modeling import ModelConfig
from italtensor.preprocessing import FeatureStandardizer


def test_generate_random_configs_is_deterministic_and_bounded():
    first = generate_random_configs(trials=24, seed=7)
    second = generate_random_configs(trials=24, seed=7)

    assert first == second
    assert len(first) == 24
    assert all(config.hidden_layers in {(16,), (32,), (64,), (64, 32)} for config in first)
    assert all(config.learning_rate in {0.01, 0.001, 0.0003} for config in first)
    assert all(config.feature_map in {"linear", "quadratic", "rff"} for config in first)
    assert {"quadratic", "rff"}.issubset({config.feature_map for config in first})


def test_generate_random_configs_rejects_more_than_search_space():
    with pytest.raises(ValueError, match="cannot exceed 216"):
        generate_random_configs(trials=217)


def test_select_best_result_ranks_f1_accuracy_then_loss():
    weak = ExperimentResult(
        config=ModelConfig(hidden_layers=(16,)),
        metrics={"f1": 0.5, "accuracy": 0.9, "validation_loss": 0.1},
        history={},
    )
    accurate = ExperimentResult(
        config=ModelConfig(hidden_layers=(32,)),
        metrics={"f1": 0.8, "accuracy": 0.7, "validation_loss": 0.4},
        history={},
    )
    best = ExperimentResult(
        config=ModelConfig(hidden_layers=(64,)),
        metrics={"f1": 0.8, "accuracy": 0.7, "validation_loss": 0.2},
        history={},
    )

    assert select_best_result([weak, accurate, best]) is best


def test_run_experiments_accepts_fake_trainer_and_reports_progress():
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.2, 0.3], [0.8, 0.9]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32)
    configs = [
        ModelConfig(hidden_layers=(16,), learning_rate=0.001),
        ModelConfig(hidden_layers=(64,), learning_rate=0.01),
    ]
    progress: list[tuple[int, int]] = []

    def fake_trainer(x_train, y_train, x_val, y_val, config):
        assert x_train.shape[1] == 2
        assert y_val.shape[0] >= 1
        score = 0.9 if config.hidden_layers == (64,) else 0.4
        return ExperimentResult(
            config=config,
            metrics={"f1": score, "accuracy": score, "validation_loss": 1.0 - score},
            history={"val_loss": [1.0 - score]},
        )

    results = run_experiments(
        features,
        labels,
        configs=configs,
        trainer=fake_trainer,
        progress_callback=lambda index, total, result: progress.append((index, total)),
    )

    assert len(results) == 2
    assert progress == [(1, 2), (2, 2)]
    assert select_best_result(results).config.hidden_layers == (64,)


def test_optimize_threshold_prefers_best_f1_threshold():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float32)

    threshold = optimize_threshold(labels, probabilities)
    metrics = evaluate_predictions(labels, probabilities, threshold)

    assert threshold == pytest.approx(0.35)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_evaluate_predictions_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions(np.asarray([0, 1]), np.asarray([0.2]))


def test_split_train_validation_requires_two_samples_per_class():
    dataset = validate_dataset([[0.0], [1.0]], [0, 1], min_samples=2, require_two_classes=True)

    with pytest.raises(ValueError, match="two samples for each class"):
        split_train_validation(dataset)


def test_balanced_class_weights_are_inverse_frequency():
    weights = balanced_class_weights(np.asarray([0, 0, 0, 1], dtype=np.int32))

    assert weights == {0: pytest.approx(2 / 3), 1: pytest.approx(2.0)}


def test_balanced_class_weights_returns_none_for_one_class():
    assert balanced_class_weights(np.asarray([1, 1, 1], dtype=np.int32)) is None


def test_permutation_feature_importance_identifies_controlling_feature():
    class FirstFeatureModel:
        def predict(self, samples, verbose=0):
            return np.asarray(samples[:, [0]], dtype=np.float32)

    features = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int32)
    importances = permutation_feature_importance(
        FirstFeatureModel(),
        FeatureStandardizer.identity(2),
        features,
        labels,
        threshold=0.5,
        max_features=2,
        repeats=6,
        seed=1,
    )

    assert importances[0]["feature_index"] == 0
    assert importances[0]["importance"] > importances[1]["importance"]


def test_platt_scaling_and_calibration_diagnostics():
    from italtensor.experiments import fit_platt_scaling, compute_ece
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    probs = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    # Test ECE calculation
    ece = compute_ece(labels, probs, n_bins=2)
    assert 0.0 <= ece <= 1.0

    # Test Platt scaling fit
    a, b = fit_platt_scaling(probs, labels)
    calibrated_logits = a * np.log(probs / (1.0 - probs)) + b
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    assert all(0.0 <= p <= 1.0 for p in calibrated_probs)


def test_stratified_kfold_indices():
    from italtensor.experiments import stratified_kfold_indices
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    splits = stratified_kfold_indices(labels, n_splits=5, seed=42)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(train_idx) == 8
        assert len(val_idx) == 2
        # Check that validation set is stratified (one 0 and one 1)
        assert np.sum(labels[val_idx] == 0) == 1
        assert np.sum(labels[val_idx] == 1) == 1


def test_train_single_model_cv():
    from italtensor.experiments import train_single_model_cv
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.2, 0.3], [0.8, 0.9], [0.1, 0.2], [0.9, 0.8]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
    config = ModelConfig(hidden_layers=(16,), learning_rate=0.01, feature_selection_k=1)
    
    result = train_single_model_cv(features, labels, config, n_splits=3)
    assert result.config.feature_selection_k == 1
    assert "cv_mean_f1" in result.metrics
    assert "cv_std_f1" in result.metrics
    assert result.metrics["cv_folds"] == 3

