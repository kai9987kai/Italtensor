import numpy as np
import pytest

from italtensor.validation_stability import (
    format_validation_stability_summary,
    run_validation_stability,
    run_validation_stability_diagnostics,
    validation_stability_dataset_fingerprint,
)


def _dataset(rows=36):
    rng = np.random.default_rng(7)
    labels = np.asarray([0, 1] * (rows // 2), dtype=np.int32)
    signed = np.where(labels == 1, 1.0, -1.0)
    features = np.column_stack(
        [
            np.arange(rows, dtype=np.float32),
            signed * 1.1 + rng.normal(0.0, 0.35, size=rows),
            signed * 0.4 + rng.normal(0.0, 0.45, size=rows),
        ]
    ).astype(np.float32)
    return features, labels


def test_validation_stability_uses_disjoint_nested_split_roles(monkeypatch):
    features, labels = _dataset(36)
    observed = []

    def fake_train_fold(
        x_train,
        y_train,
        x_calibration,
        y_calibration,
        x_evaluation,
        y_evaluation,
        *,
        feature_map,
        max_epochs,
        seed,
    ):
        train_ids = set(x_train[:, 0].astype(int).tolist())
        calibration_ids = set(x_calibration[:, 0].astype(int).tolist())
        evaluation_ids = set(x_evaluation[:, 0].astype(int).tolist())
        observed.append((train_ids, calibration_ids, evaluation_ids))
        calibration_probabilities = 1.0 / (1.0 + np.exp(-3.0 * x_calibration[:, 1]))
        evaluation_probabilities = 1.0 / (1.0 + np.exp(-3.0 * x_evaluation[:, 1]))
        return evaluation_probabilities, calibration_probabilities, {"loss": [0.4], "val_loss": [0.3]}

    monkeypatch.setattr("italtensor.validation_stability._train_fold", fake_train_fold)

    report = run_validation_stability_diagnostics(
        features,
        labels,
        n_splits=3,
        repeats=2,
        max_epochs=5,
        seed=11,
    )

    assert report["total_fold_count"] == 6
    assert len(observed) == 6
    for train_ids, calibration_ids, evaluation_ids in observed:
        assert train_ids.isdisjoint(calibration_ids)
        assert train_ids.isdisjoint(evaluation_ids)
        assert calibration_ids.isdisjoint(evaluation_ids)
        assert train_ids | calibration_ids | evaluation_ids == set(range(36))
    assert report["aggregate"]["f1"]["mean"] > 0.80
    assert report["interpretation_note"].endswith("not formal confidence intervals.")
    assert report["folds"][0]["calibration_sample_count"] > 0
    assert report["folds"][0]["tuned_threshold"] >= 0.0


def test_validation_stability_is_deterministic_with_real_numpy_models():
    features, labels = _dataset(40)
    kwargs = {
        "n_splits": 4,
        "repeats": 2,
        "max_epochs": 8,
        "feature_map": "linear",
        "seed": 5,
    }

    first = run_validation_stability(features, labels, **kwargs)
    second = run_validation_stability(features, labels, **kwargs)

    assert first == second
    assert first["summary"]["verdict"] in {
        "validation_stable",
        "validation_stability_review",
        "validation_unstable",
    }
    assert first["aggregate"]["f1"]["std"] >= 0.0
    assert first["tuned_aggregate"]["f1"]["mean"] >= 0.0
    assert first["calibration_threshold_distribution"]["std"] >= 0.0
    assert format_validation_stability_summary(first).startswith("Validation stability:")


def test_validation_stability_rejects_bad_inputs():
    features, labels = _dataset(36)

    with pytest.raises(ValueError, match="2D"):
        run_validation_stability([0.1, 0.2], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        bad = features.copy()
        bad[0, 1] = np.nan
        run_validation_stability(bad, labels)
    with pytest.raises(ValueError, match="binary labels"):
        run_validation_stability(features, labels[:-1].tolist() + [2])
    with pytest.raises(ValueError, match="counts do not match"):
        run_validation_stability(features, labels[:-1])
    with pytest.raises(ValueError, match="between 2 and 10"):
        run_validation_stability(features, labels, n_splits=1)
    with pytest.raises(ValueError, match="between 1 and 20"):
        run_validation_stability(features, labels, repeats=0)
    with pytest.raises(ValueError, match="feature_map"):
        run_validation_stability(features, labels, feature_map="bad-map")
    with pytest.raises(ValueError, match="threshold"):
        run_validation_stability(features, labels, threshold=1.2)


def test_validation_stability_fingerprint_is_order_sensitive():
    features, labels = _dataset(36)

    left = validation_stability_dataset_fingerprint(features, labels)
    right = validation_stability_dataset_fingerprint(features[::-1], labels[::-1])

    assert left != right
