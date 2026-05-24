from __future__ import annotations

import numpy as np

from italtensor.cartography import format_cartography_summary, run_dataset_cartography
from italtensor.modeling import ModelConfig, train_numpy_model


def test_dataset_cartography_regions():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(24, 3)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.int32)
    config = ModelConfig(max_epochs=20, feature_map="linear", random_seed=1)
    model, _ = train_numpy_model(features, labels, config)
    report = run_dataset_cartography(model, features, labels, n_perturbations=3)
    assert report["sample_count"] == 24
    assert sum(report["region_counts"].values()) == 24
    assert "easy_to_learn" in report["regions"]
    summary = format_cartography_summary(report)
    assert "Dataset cartography" in summary
