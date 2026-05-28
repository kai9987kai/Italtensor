from italtensor.trial_inspector import format_trial_inspector_summary, inspect_trial_history


def _trial(
    *,
    backend: str = "numpy",
    feature_map: str = "linear",
    hidden_layers: list[int] | None = None,
    f1: float = 0.5,
    accuracy: float = 0.5,
    loss: float = 0.8,
    brier: float | None = None,
    ece: float | None = None,
) -> dict:
    return {
        "config": {
            "backend": backend,
            "feature_map": feature_map,
            "hidden_layers": hidden_layers or [32],
            "learning_rate": 0.001,
            "batch_size": 16,
            "max_epochs": 50,
        },
        "metrics": {
            "f1": f1,
            "accuracy": accuracy,
            "validation_loss": loss,
            "brier_score": brier if brier is not None else 0.18,
            "ece": ece if ece is not None else 0.04,
        },
        "threshold": 0.45,
    }


def test_trial_inspector_recommends_creating_history_when_empty():
    report = inspect_trial_history([])

    assert report["valid_trial_count"] == 0
    assert report["summary"]["best_trial_index"] is None
    assert report["recommendations"][0]["category"] == "search"
    assert "Run Train once" in report["summary"]["recommendation"]


def test_trial_inspector_ranks_by_f1_accuracy_then_lower_loss():
    report = inspect_trial_history(
        [
            _trial(backend="numpy", feature_map="linear", f1=0.72, accuracy=0.75, loss=0.40),
            _trial(backend="mps", feature_map="linear", f1=0.72, accuracy=0.78, loss=0.42),
            _trial(backend="keras", feature_map="rff", f1=0.72, accuracy=0.78, loss=0.35),
            _trial(backend="numpy", feature_map="rff", f1=0.68, accuracy=0.82, loss=0.30),
        ]
    )

    best = report["leaderboard"][0]
    assert best["trial_index"] == 3
    assert best["backend"] == "keras"
    assert best["validation_loss"] == 0.35
    assert report["summary"]["leader_margin_f1"] == 0.0
    assert any(item["category"] == "stability" for item in report["recommendations"])


def test_trial_inspector_groups_backend_and_feature_map_families():
    report = inspect_trial_history(
        [
            _trial(backend="numpy", feature_map="rff", f1=0.81, accuracy=0.82, loss=0.32),
            _trial(backend="numpy", feature_map="rff", hidden_layers=[64], f1=0.78, accuracy=0.80, loss=0.34),
            _trial(backend="mps", feature_map="linear", f1=0.61, accuracy=0.64, loss=0.55),
            _trial(backend="keras", feature_map="linear", f1=0.58, accuracy=0.62, loss=0.60),
        ]
    )

    assert report["groups"][0]["group"] == "numpy/rff"
    assert report["groups"][0]["count"] == 2
    assert report["summary"]["winner_stability"]["group"] == "numpy/rff"
    assert report["summary"]["search_diversity"]["backend_count"] == 3
    assert report["summary"]["search_diversity"]["feature_map_count"] == 2


def test_trial_inspector_flags_malformed_thin_and_calibration_risk():
    report = inspect_trial_history(
        [
            _trial(f1=0.55, accuracy=0.85, loss=0.50, brier=0.26, ece=0.12),
            {"config": {"backend": "numpy"}, "metrics": {"f1": float("nan")}},
        ]
    )
    categories = {item["category"] for item in report["recommendations"]}

    assert report["invalid_trial_count"] == 1
    assert "search" in categories
    assert "calibration" in categories
    assert "model_selection" in categories
    assert "metric" in report["summary"]["warning"] or "malformed" in report["summary"]["warning"]
    summary = format_trial_inspector_summary(report)
    assert summary.startswith("Trial inspector:")
    assert "valid_trials=1" in summary
