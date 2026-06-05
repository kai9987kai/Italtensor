import numpy as np
import pytest

from italtensor.label_noise_stress import (
    format_label_noise_stress_summary,
    label_noise_dataset_fingerprint,
    label_noise_stress_dataset_fingerprint,
    run_label_noise_stress,
    run_label_noise_stress_diagnostics,
)
from italtensor.presets import generate_builtin_preset


def test_label_noise_stress_accepts_numpy_arrays_and_reports_degradation():
    dataset = generate_builtin_preset("Label noise stress lab", sample_count=80, seed=7)

    report = run_label_noise_stress_diagnostics(
        dataset.features,
        dataset.labels,
        noise_rates=(0.0, 0.10, 0.25),
        repeats=2,
        max_epochs=12,
        seed=3,
    )

    assert report["sample_count"] == 80
    assert report["input_dim"] == 4
    assert report["baseline"]["mean_metrics"]["f1"] >= 0.90
    assert [item["noise_rate"] for item in report["rates"]] == [0.0, 0.1, 0.25]
    assert report["rates"][0]["repeat_count"] == 1
    assert report["rates"][2]["repeat_count"] == 2
    assert report["summary"]["worst_mean_f1_drop"] > 0.0
    assert report["summary"]["priority"] in {"low", "medium", "high"}
    assert report["summary"]["recommended_next_step"]

    summary = format_label_noise_stress_summary(report)
    assert summary.startswith("Label noise stress:")
    assert "worst_f1_drop=" in summary


def test_label_noise_stress_alias_and_fingerprint_are_order_sensitive():
    features = np.asarray([[-2.0], [-1.6], [-1.1], [1.0], [1.5], [2.0]] * 2, dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1] * 2, dtype=np.int32)

    report = run_label_noise_stress(
        features,
        labels,
        noise_rates=(0.0, 0.20),
        repeats=1,
        max_epochs=6,
        seed=2,
    )
    left = label_noise_stress_dataset_fingerprint(features, labels)
    right = label_noise_dataset_fingerprint(features[::-1], labels[::-1])

    assert report["dataset_fingerprint"] == left
    assert left != right


def test_label_noise_stress_rejects_bad_inputs():
    features = [[-1.0], [-0.8], [-0.6], [-0.4], [-0.2], [-0.1], [0.1], [0.2], [0.4], [0.6], [0.8], [1.0]]
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    with pytest.raises(ValueError, match="2D"):
        run_label_noise_stress([0.1, 0.2], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        run_label_noise_stress([[0.1], [np.nan]] * 6, labels)
    with pytest.raises(ValueError, match="binary labels"):
        run_label_noise_stress(features, labels[:-1] + [2])
    with pytest.raises(ValueError, match="counts do not match"):
        run_label_noise_stress(features, labels[:-1])
    with pytest.raises(ValueError, match="at least 12"):
        run_label_noise_stress(features[:10], labels[:10])
    with pytest.raises(ValueError, match="between 0.0 and 0.5"):
        run_label_noise_stress(features, labels, noise_rates=(0.0, 0.8))
    with pytest.raises(ValueError, match="feature_map"):
        run_label_noise_stress(features, labels, feature_map="bad-map")
