import numpy as np
import pytest

from italtensor.modeling import ModelConfig
from italtensor.automl import HyperparameterSpace, TPEOptimizer, gaussian_kde_density, estimate_bandwidth, run_auto_tune

def test_hyperparameter_space():
    space = HyperparameterSpace(n_features=10)
    rng = np.random.default_rng(42)
    
    # Test random sampling
    params = space.sample_random(rng)
    assert "learning_rate" in params
    assert 1e-4 <= params["learning_rate"] <= 1e-1
    assert "l1_penalty" in params
    assert 0.0 <= params["l1_penalty"] <= 0.1
    assert "max_epochs" in params
    assert params["max_epochs"] in [10, 20, 50, 100]
    assert "batch_size" in params
    assert params["batch_size"] in [8, 16, 32, 64]
    assert "feature_selection_k" in params
    assert params["feature_selection_k"] in [None, 2, 5]


def test_kde_and_bandwidth():
    samples = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32)
    bw = estimate_bandwidth(samples)
    assert bw > 0.0
    
    density = gaussian_kde_density(1.2, samples, bw)
    assert density > 0.0
    
    # Density should be higher near the mean/center than far away
    density_far = gaussian_kde_density(10.0, samples, bw)
    assert density > density_far


def test_tpe_optimizer_suggestions():
    space = HyperparameterSpace(n_features=5)
    optimizer = TPEOptimizer(space, n_startup=3, seed=42)
    
    # Startup trials should be random
    s1 = optimizer.suggest()
    optimizer.register_trial(s1, 0.5)
    s2 = optimizer.suggest()
    optimizer.register_trial(s2, 0.8)
    s3 = optimizer.suggest()
    optimizer.register_trial(s3, 0.2)
    
    # 4th trial should run TPE suggestion
    s4 = optimizer.suggest()
    assert isinstance(s4, dict)
    assert "learning_rate" in s4


def test_run_auto_tune_smoke():
    # Simple synthetic classification dataset
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 4))
    # Simple rule: if first feature is positive, label is 1
    labels = (features[:, 0] > 0.0).astype(np.int32)
    
    base_config = ModelConfig(
        max_epochs=2,
        batch_size=8,
        random_seed=42,
    )
    
    logs = []
    def logger_cb(msg):
        logs.append(msg)
        
    best_params, best_result = run_auto_tune(
        features,
        labels,
        n_trials=3,
        base_config=base_config,
        use_cv=False,
        tune_lr=True,
        tune_l1=True,
        tune_smote=True,
        tune_features=True,
        logger=logger_cb,
    )
    
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "l1_penalty" in best_params
    assert "gradient_clip" in best_params
    assert "max_epochs" in best_params
    assert "batch_size" in best_params
    assert "feature_selection_k" in best_params
    assert "use_smote" in best_params
    assert "smote_k" in best_params
    
    assert best_result is not None
    assert len(logs) > 0
    assert any("AutoML Tuning Sweep Completed Successfully!" in line for line in logs)
