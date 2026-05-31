import numpy as np
import pytest

from italtensor.capacity_planner import (
    capacity_planner_dataset_fingerprint,
    format_capacity_planner_summary,
    run_capacity_planner,
)
from italtensor.preprocessing import FeatureStandardizer


class ProbabilityModel:
    def predict(self, samples, verbose=0):
        return np.asarray(samples, dtype=np.float32)[:, :1]


class ShortModel:
    def predict(self, samples, verbose=0):
        return np.asarray([[0.2]], dtype=np.float32)


class NanModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), np.nan, dtype=np.float32)


class OutOfRangeModel:
    def predict(self, samples, verbose=0):
        return np.full((np.asarray(samples).shape[0], 1), 1.2, dtype=np.float32)


def test_capacity_planner_ranks_budget_points_and_rows():
    features = np.asarray([[0.95], [0.80], [0.70], [0.30], [0.20], [0.10]], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int32)

    report = run_capacity_planner(
        ProbabilityModel(),
        features,
        labels,
        capacity_grid=[1 / 6, 0.5, 1.0],
        benefit_tp=5.0,
        cost_fp=3.0,
        cost_action=0.25,
    )

    assert report["summary"]["verdict"] == "actionable_capacity_plan"
    assert report["best_utility"]["k"] == 3
    assert report["best_utility"]["true_positive"] == 2
    assert report["best_utility"]["precision_at_k"] == pytest.approx(2 / 3)
    assert report["best_utility"]["recall_captured"] == pytest.approx(2 / 3)
    assert report["capacity_points"] == report["points"]
    assert report["top_rows"][0]["row_index"] == 0
    assert report["dataset_fingerprint"] == capacity_planner_dataset_fingerprint(features, labels)
    assert format_capacity_planner_summary(report).startswith("Capacity planner:")


def test_capacity_planner_uses_selected_preprocessor_once():
    features = np.asarray([[999.0, 0.95], [999.0, 0.20], [999.0, 0.80], [999.0, 0.10]], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int32)
    preprocessor = FeatureStandardizer(
        mean=np.asarray([0.0], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        selected_indices=[1],
    )

    report = run_capacity_planner(ProbabilityModel(), features, labels, preprocessor=preprocessor, capacity_grid=[0.5])

    assert report["sample_count"] == 4
    assert report["input_dim"] == 2
    assert report["points"][0]["precision_at_k"] == pytest.approx(1.0)


def test_capacity_planner_collapses_duplicate_budgets_and_tie_breaks_by_row_index():
    features = np.asarray([[0.50], [0.50], [0.50], [0.20]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)

    report = run_capacity_planner(
        ProbabilityModel(),
        features,
        labels,
        capacity_grid=[0.01, 0.02, 0.25, 0.50],
        max_rows=4,
    )

    assert [point["k"] for point in report["points"]] == [1, 2]
    assert [row["row_index"] for row in report["top_rows"][:3]] == [0, 1, 2]


def test_capacity_planner_fingerprint_is_order_sensitive():
    features = np.asarray([[0.1], [0.9], [0.2], [0.8]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    order = [2, 0, 3, 1]

    assert capacity_planner_dataset_fingerprint(features, labels) != capacity_planner_dataset_fingerprint(
        features[order],
        labels[order],
    )


def test_capacity_planner_validates_inputs_and_probabilities():
    features = np.asarray([[0.1], [0.9]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="different number"):
        run_capacity_planner(ShortModel(), features, labels)
    with pytest.raises(ValueError, match="finite"):
        run_capacity_planner(NanModel(), features, labels)
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_capacity_planner(OutOfRangeModel(), features, labels)
    with pytest.raises(ValueError, match="binary"):
        run_capacity_planner(ProbabilityModel(), features, np.asarray([0.5, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite numbers"):
        run_capacity_planner(ProbabilityModel(), np.asarray([[0.1], [np.inf]], dtype=np.float32), labels)


def test_capacity_planner_handles_one_class_and_one_row_datasets():
    no_positive = run_capacity_planner(
        ProbabilityModel(),
        np.asarray([[0.8], [0.7], [0.2]], dtype=np.float32),
        np.asarray([0, 0, 0], dtype=np.int32),
        capacity_grid=[0.5, 1.0],
    )
    one_row = run_capacity_planner(
        ProbabilityModel(),
        np.asarray([[0.8]], dtype=np.float32),
        np.asarray([1], dtype=np.int32),
        capacity_grid=[0.1],
    )

    assert no_positive["summary"]["verdict"] == "no_positive_evidence"
    assert no_positive["summary"]["best_recall_captured"] == 0.0
    assert one_row["summary"]["best_k"] == 1
    assert one_row["points"][0]["capacity_fraction"] == 1.0
