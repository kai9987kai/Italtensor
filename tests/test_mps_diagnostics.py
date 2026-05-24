from __future__ import annotations

import numpy as np

from italtensor.mps_diagnostics import format_mps_sweep_summary, run_mps_bond_sweep
from italtensor.modeling import ModelConfig


def test_mps_bond_sweep_ranks_dimensions():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(40, 4)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int32)
    config = ModelConfig(max_epochs=8, batch_size=8, mps_bond_dim=8, backend="mps", random_seed=3)
    report = run_mps_bond_sweep(features, labels, config, bond_dims=(4, 8))
    assert len(report["results"]) == 2
    assert report["recommended_bond_dim"] in (4, 8)
    summary = format_mps_sweep_summary(report)
    assert "MPS bond sweep" in summary
