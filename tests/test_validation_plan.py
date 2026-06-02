from __future__ import annotations

import numpy as np

from italtensor.validation_plan import (
    format_validation_plan_summary,
    run_validation_plan,
    validation_plan_dataset_fingerprint,
)


def test_validation_plan_collects_more_labels_for_one_class():
    features = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    labels = [0, 0, 0]

    report = run_validation_plan(features, labels)

    assert report["summary"]["recommended_strategy"] == "collect_more_labels"
    assert report["summary"]["risk_level"] == "high"
    assert any(check["name"] == "two_classes" and check["status"] == "fail" for check in report["checks"])
    assert "Validation plan" in format_validation_plan_summary(report)


def test_validation_plan_recommends_stratified_kfold_for_small_balanced_data():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(24, 3)).astype(np.float32)
    labels = np.asarray([0, 1] * 12, dtype=np.int32)

    report = run_validation_plan(features, labels)

    assert report["summary"]["recommended_strategy"] == "stratified_kfold"
    assert report["summary"]["kfold_splits"] == 5
    assert report["split_blueprint"]["stratify"] is True
    assert report["split_blueprint"]["shuffle"] is True


def test_validation_plan_recommends_holdout_for_larger_balanced_data():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(120, 4)).astype(np.float32)
    labels = np.asarray([0, 1] * 60, dtype=np.int32)

    report = run_validation_plan(features, labels)

    assert report["summary"]["recommended_strategy"] == "stratified_holdout"
    assert report["summary"]["validation_fraction"] == 0.2
    assert report["summary"]["risk_level"] == "low"


def test_validation_plan_detects_ordered_drift():
    rng = np.random.default_rng(4)
    first = rng.normal(loc=0.0, scale=0.4, size=(30, 3))
    second = rng.normal(loc=(1.2, 0.0, 0.0), scale=0.4, size=(30, 3))
    features = np.vstack([first, second]).astype(np.float32)
    labels = np.asarray([0] * 22 + [1] * 8 + [0] * 8 + [1] * 22, dtype=np.int32)

    report = run_validation_plan(features, labels)

    assert report["summary"]["recommended_strategy"] == "chronological_holdout"
    assert report["summary"]["row_order_risk"] is True
    assert report["split_blueprint"]["shuffle"] is False
    assert report["row_order"]["top_shift_feature"] == 0


def test_validation_plan_fingerprint_is_order_sensitive():
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.3], [0.3, 0.2]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    original = validation_plan_dataset_fingerprint(features, labels)
    shuffled = validation_plan_dataset_fingerprint(features[::-1], labels[::-1])

    assert original != shuffled
