from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import Dataset, DataValidationError, dataset_from_jsonable, dataset_to_jsonable, validate_dataset

SCHEMA_VERSION = 1
DEFAULT_TRAINING_DEFAULTS = {"epochs": 50, "batch_size": 16, "trials": 8, "feature_map": "linear"}


@dataclass(frozen=True)
class PresetInfo:
    key: str
    name: str
    description: str
    default_samples: int
    min_samples: int = 4
    input_dim: int = 2
    recommended_feature_map: str = "linear"
    feature_names: tuple[str, ...] = ("x1", "x2")
    label_names: tuple[str, str] = ("negative", "positive")
    training_defaults: dict[str, object] = field(default_factory=lambda: DEFAULT_TRAINING_DEFAULTS.copy())
    prediction_examples: tuple[dict[str, object], ...] = field(default_factory=tuple)


BUILT_IN_PRESETS: tuple[PresetInfo, ...] = (
    PresetInfo(
        key="linear_blobs",
        name="Linear blobs",
        description="Two separable 2D Gaussian clusters for quick sanity checks.",
        default_samples=80,
        recommended_feature_map="linear",
        prediction_examples=(
            {"name": "Likely class 0", "features": [-1.25, -1.0], "expected_label": 0},
            {"name": "Likely class 1", "features": [1.25, 1.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="xor",
        name="XOR pattern",
        description="A nonlinear 2D pattern that rewards hidden layers over a linear boundary.",
        default_samples=96,
        recommended_feature_map="quadratic",
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Low-low", "features": [-1.0, -1.0], "expected_label": 0},
            {"name": "Low-high", "features": [-1.0, 1.0], "expected_label": 1},
            {"name": "High-low", "features": [1.0, -1.0], "expected_label": 1},
            {"name": "High-high", "features": [1.0, 1.0], "expected_label": 0},
        ),
    ),
    PresetInfo(
        key="imbalanced_blobs",
        name="Imbalanced blobs",
        description="A skewed 2D binary dataset for testing class weights and balanced metrics.",
        default_samples=100,
        recommended_feature_map="linear",
        training_defaults={"epochs": 60, "batch_size": 16, "trials": 10, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Common region", "features": [-0.35, -0.1], "expected_label": 0},
            {"name": "Minority region", "features": [1.15, 1.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="signal_plus_noise",
        name="Signal plus noise",
        description="Six features where only the first two drive the label, useful for feature importance.",
        default_samples=120,
        input_dim=6,
        recommended_feature_map="linear",
        feature_names=("signal_a", "signal_b", "noise_1", "noise_2", "noise_3", "noise_4"),
        training_defaults={"epochs": 60, "batch_size": 16, "trials": 10, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Lower score", "features": [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "expected_label": 0},
            {"name": "Higher score", "features": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="concentric_rings",
        name="Concentric rings",
        description="A radial nonlinear dataset for trying RFF feature maps.",
        default_samples=120,
        recommended_feature_map="rff",
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 16, "feature_map": "rff"},
        prediction_examples=(
            {"name": "Inner ring", "features": [0.65, 0.0], "expected_label": 0},
            {"name": "Outer ring", "features": [1.35, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="two_moons",
        name="Two moons",
        description="Interleaving crescent shapes for nonlinear boundary experiments.",
        default_samples=120,
        recommended_feature_map="rff",
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 16, "feature_map": "rff"},
        prediction_examples=(
            {"name": "Upper arc", "features": [0.0, 1.0], "expected_label": 0},
            {"name": "Lower arc", "features": [1.0, -0.55], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="rare_event_signal",
        name="Rare event signal",
        description="A heavily imbalanced dataset with a compact positive region.",
        default_samples=160,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("region_x", "region_y", "background_1", "background_2"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Typical negative", "features": [0.0, 0.0, 0.0, 0.0], "expected_label": 0},
            {"name": "Rare positive", "features": [1.7, 1.5, 0.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="overlapping_margin",
        name="Overlapping margin",
        description="Partly overlapping diagonal clusters for uncertainty and abstention experiments.",
        default_samples=140,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Likely class 0", "features": [-0.85, -0.7], "expected_label": 0},
            {"name": "Ambiguous margin", "features": [0.05, 0.0], "expected_label": None},
            {"name": "Likely class 1", "features": [0.85, 0.7], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="noisy_labels",
        name="Noisy labels",
        description="Mostly separable diagonal blobs with controlled label flips for robustness checks.",
        default_samples=140,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clean negative region", "features": [-1.1, -0.9], "expected_label": 0},
            {"name": "Clean positive region", "features": [1.1, 0.9], "expected_label": 1},
            {"name": "Ambiguous noisy margin", "features": [0.0, 0.0], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="sparse_interaction_signal",
        name="Sparse interaction signal",
        description="Sixteen features where a few sparse terms and one interaction drive the label.",
        default_samples=180,
        input_dim=16,
        recommended_feature_map="quadratic",
        feature_names=tuple(f"feature_{index + 1}" for index in range(16)),
        training_defaults={
            "epochs": 90,
            "batch_size": 16,
            "trials": 16,
            "feature_map": "quadratic",
            "l1_penalty": 0.001,
            "feature_selection_k": 6,
        },
        prediction_examples=(
            {
                "name": "Sparse negative",
                "features": [-1.0, 0.4, 0.0, -0.8, 0.0, 0.0, 0.0, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "expected_label": 0,
            },
            {
                "name": "Interaction positive",
                "features": [1.0, 1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "expected_label": 1,
            },
        ),
    ),
    PresetInfo(
        key="deployment_drift_probe",
        name="Deployment drift probe",
        description="Compact training distribution with examples that make shifted batch rows easy to flag.",
        default_samples=140,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("signal_x", "signal_y", "stable_noise", "shift_probe"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 10, "feature_map": "linear"},
        prediction_examples=(
            {"name": "In-distribution negative", "features": [-0.9, -0.7, 0.0, 0.0], "expected_label": 0},
            {"name": "In-distribution positive", "features": [0.9, 0.7, 0.0, 0.0], "expected_label": 1},
            {"name": "Drift review row", "features": [0.0, 0.0, 4.5, -4.5], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="active_learning_margin",
        name="Active learning margin",
        description="Dense linear classes with an ambiguous boundary belt for query-ranking and counterfactual recourse.",
        default_samples=160,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clear negative", "features": [-1.2, -0.55], "expected_label": 0},
            {"name": "Boundary query", "features": [0.03, -0.02], "expected_label": None},
            {"name": "Clear positive", "features": [1.2, 0.55], "expected_label": 1},
            {"name": "Drifting candidate", "features": [0.0, 3.8], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="spurious_shortcut",
        name="Spurious shortcut",
        description="A strong shortcut feature works during training but invites robustness and drift stress tests.",
        default_samples=180,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("stable_signal", "context_noise", "shortcut_signal"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Shortcut negative", "features": [-0.7, 0.0, -1.2], "expected_label": 0},
            {"name": "Shortcut positive", "features": [0.7, 0.0, 1.2], "expected_label": 1},
            {"name": "Shortcut conflict", "features": [0.8, 0.0, -1.2], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="subgroup_blind_spot",
        name="Subgroup blind spot",
        description="A minority subgroup follows a different rule, useful for slice diagnostics and interaction models.",
        default_samples=180,
        input_dim=3,
        recommended_feature_map="quadratic",
        feature_names=("primary_signal", "subgroup_marker", "context_noise"),
        training_defaults={"epochs": 90, "batch_size": 16, "trials": 16, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Majority negative", "features": [-0.8, 0.0, 0.0], "expected_label": 0},
            {"name": "Majority positive", "features": [0.8, 0.0, 0.0], "expected_label": 1},
            {"name": "Minority flipped rule", "features": [0.8, 1.0, 0.0], "expected_label": 0},
        ),
    ),
    PresetInfo(
        key="subgroup_disparity_lab",
        name="Subgroup disparity lab",
        description="A visible marker group has different error behavior for subgroup disparity diagnostics.",
        default_samples=220,
        input_dim=3,
        recommended_feature_map="quadratic",
        feature_names=("primary_signal", "group_marker", "context_noise"),
        training_defaults={"epochs": 90, "batch_size": 16, "trials": 16, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Majority clear positive", "features": [1.0, 0.0, 0.0], "expected_label": 1},
            {"name": "Minority conflict", "features": [1.0, 1.0, 0.0], "expected_label": None},
            {"name": "Majority clear negative", "features": [-1.0, 0.0, 0.0], "expected_label": 0},
        ),
    ),
    PresetInfo(
        key="response_curve_lab",
        name="Response curve lab",
        description="Monotonic, nonlinear, and weak features for model-response and partial-dependence diagnostics.",
        default_samples=220,
        input_dim=3,
        recommended_feature_map="quadratic",
        feature_names=("linear_driver", "nonlinear_arc", "weak_noise"),
        training_defaults={"epochs": 90, "batch_size": 16, "trials": 16, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Low response", "features": [-1.2, -1.0, 0.0], "expected_label": 0},
            {"name": "Arc peak", "features": [0.1, 0.0, 0.0], "expected_label": 1},
            {"name": "Arc tail", "features": [0.4, 1.4, 0.0], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="interaction_surface_lab",
        name="Interaction surface lab",
        description="A product-like x1*x2 signal with distractors for pairwise interaction diagnostics.",
        default_samples=240,
        input_dim=4,
        recommended_feature_map="quadratic",
        feature_names=("left_factor", "right_factor", "weak_signal", "distractor_noise"),
        training_defaults={"epochs": 90, "batch_size": 16, "trials": 16, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Both factors aligned", "features": [1.2, 1.2, 0.0, 0.0], "expected_label": 1},
            {"name": "Opposing factors", "features": [1.2, -1.2, 0.0, 0.0], "expected_label": 0},
            {"name": "Weak-signal distractor", "features": [0.1, 0.1, 1.0, 0.0], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="calibration_repair_lab",
        name="Calibration repair lab",
        description="Overconfident margins and noisy shoulders for post-hoc probability calibration experiments.",
        default_samples=220,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("margin_score", "confidence_trap", "background_noise"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Confident negative", "features": [-1.2, 1.0, 0.0], "expected_label": 0},
            {"name": "Miscalibrated shoulder", "features": [0.35, 1.0, 0.0], "expected_label": None},
            {"name": "Confident positive", "features": [1.2, 1.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="permutation_null_lab",
        name="Permutation null lab",
        description="A real margin signal with weak support and decoys for shuffled-label significance checks.",
        default_samples=220,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("real_margin", "weak_support", "label_noise_marker", "decoy_noise"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clear null-negative", "features": [-1.2, -0.4, 0.0, 0.0], "expected_label": 0},
            {"name": "Boundary null check", "features": [0.02, 0.0, 1.0, 0.0], "expected_label": None},
            {"name": "Clear null-positive", "features": [1.2, 0.4, 0.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="population_drift_lab",
        name="Population drift lab",
        description="Ordered reference/current rows with shifted features for population drift monitoring.",
        default_samples=240,
        min_samples=6,
        input_dim=5,
        recommended_feature_map="linear",
        feature_names=("stable_signal", "shifted_signal", "variance_drift", "tail_probe", "decoy_noise"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Reference-like negative", "features": [-0.8, 0.0, 0.0, 0.0, 0.0], "expected_label": 0},
            {"name": "Current shifted review", "features": [0.1, 1.5, 1.8, 3.0, 0.0], "expected_label": None},
            {"name": "Reference-like positive", "features": [0.8, 0.0, 0.0, 0.0, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="adversarial_validation_lab",
        name="Adversarial validation lab",
        description="Ordered reference/current rows with multivariate drift for domain-classifier detection.",
        default_samples=240,
        min_samples=12,
        input_dim=5,
        recommended_feature_map="quadratic",
        feature_names=("stable_signal", "domain_shift_axis", "interaction_shift", "variance_marker", "decoy_noise"),
        training_defaults={"epochs": 90, "batch_size": 16, "trials": 12, "feature_map": "quadratic"},
        prediction_examples=(
            {"name": "Reference pattern", "features": [0.4, 0.6, 0.6, 0.0, 0.0], "expected_label": 1},
            {"name": "Current multivariate shift", "features": [0.4, 0.6, -0.6, 1.4, 0.0], "expected_label": None},
            {"name": "Reference negative", "features": [-0.6, -0.4, -0.4, 0.0, 0.0], "expected_label": 0},
        ),
    ),
    PresetInfo(
        key="chronological_holdout_lab",
        name="Chronological holdout lab",
        description="Ordered early/current rows with a changing label rule for temporal replay diagnostics.",
        default_samples=240,
        min_samples=16,
        input_dim=5,
        recommended_feature_map="linear",
        feature_names=("early_signal", "late_signal_decay", "threshold_drift", "calibration_wobble", "decoy_noise"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Early stable negative", "features": [-1.0, -0.8, -0.2, 0.0, 0.0], "expected_label": 0},
            {"name": "Early stable positive", "features": [1.0, 0.8, 0.2, 0.0, 0.0], "expected_label": 1},
            {"name": "Late degraded review", "features": [0.4, 1.1, 1.4, 0.8, 0.0], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="cost_sensitive_screening",
        name="Cost-sensitive screening",
        description="Rare positives with overlapping scores for threshold tradeoff and false-negative-cost experiments.",
        default_samples=180,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("risk_score", "secondary_signal", "background_noise"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Likely negative", "features": [-0.6, -0.2, 0.0], "expected_label": 0},
            {"name": "Borderline review", "features": [0.25, 0.1, 0.0], "expected_label": None},
            {"name": "Likely positive", "features": [1.0, 0.7, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="decision_utility_tradeoff",
        name="Decision utility tradeoff",
        description="Rare positives with a broad gray zone for decision-curve net-benefit experiments.",
        default_samples=220,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("risk_score", "triage_signal", "noise_marker", "gray_zone"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Avoid action", "features": [-0.8, -0.4, 0.0, -0.5], "expected_label": 0},
            {"name": "Decision gray zone", "features": [0.25, 0.15, 0.0, 0.25], "expected_label": None},
            {"name": "Act despite cost", "features": [1.1, 0.8, 0.0, 0.6], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="selective_abstention_triage",
        name="Selective abstention triage",
        description="Easy extremes plus ambiguous boundary rows for risk-coverage and abstention experiments.",
        default_samples=200,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("triage_score", "support_signal", "ambiguity_marker"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Confident negative", "features": [-1.1, -0.6, -0.8], "expected_label": 0},
            {"name": "Abstention candidate", "features": [0.02, 0.0, 1.0], "expected_label": None},
            {"name": "Confident positive", "features": [1.1, 0.6, -0.8], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="conformal_coverage_lab",
        name="Conformal coverage lab",
        description="Confident cores plus overlapping shoulders for split-conformal coverage and prediction-set experiments.",
        default_samples=220,
        input_dim=3,
        recommended_feature_map="linear",
        feature_names=("score_signal", "calibration_noise", "overlap_band"),
        training_defaults={"epochs": 80, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Singleton negative", "features": [-1.25, -0.35, -0.8], "expected_label": 0},
            {"name": "Both-label set candidate", "features": [0.0, 0.1, 1.0], "expected_label": None},
            {"name": "Singleton positive", "features": [1.25, 0.35, -0.8], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="label_audit_traps",
        name="Label audit traps",
        description="Mostly clean separable classes with a small set of flipped labels for sample-review drills.",
        default_samples=160,
        recommended_feature_map="linear",
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clean negative", "features": [-1.3, -1.0], "expected_label": 0},
            {"name": "Clean positive", "features": [1.3, 1.0], "expected_label": 1},
            {"name": "Suspicious positive-shaped negative", "features": [1.2, 0.9], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="ood_sentinel_lab",
        name="OOD sentinel lab",
        description="Mostly ordinary rows plus leverage and artifact rows for outlier screening.",
        default_samples=180,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("main_signal", "support_signal", "leverage_axis", "artifact_code"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Typical negative", "features": [-1.0, -0.6, 0.0, 0.0], "expected_label": 0},
            {"name": "Typical positive", "features": [1.0, 0.6, 0.0, 0.0], "expected_label": 1},
            {"name": "Leverage review row", "features": [0.2, 0.1, 5.0, -4.0], "expected_label": None},
        ),
    ),
    PresetInfo(
        key="bootstrap_stability_lab",
        name="Bootstrap stability lab",
        description="Stable cores plus boundary rows where resampled committees should disagree.",
        default_samples=220,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("stable_margin", "support_signal", "boundary_band", "decoy_noise"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Stable negative", "features": [-1.2, -0.5, -0.8, 0.0], "expected_label": 0},
            {"name": "Unstable boundary", "features": [0.02, 0.0, 1.1, 0.0], "expected_label": None},
            {"name": "Stable positive", "features": [1.2, 0.5, -0.8, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="prototype_coverage_lab",
        name="Prototype coverage lab",
        description="Dense class cores, boundary bridges, and sparse islands for nearest-neighbor prototype audits.",
        default_samples=220,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("coverage_axis", "class_axis", "boundary_bridge", "island_marker"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Class 0 prototype", "features": [-1.2, -0.8, -0.8, 0.0], "expected_label": 0},
            {"name": "Boundary bridge", "features": [0.0, 0.0, 1.0, 0.0], "expected_label": None},
            {"name": "Sparse island review", "features": [2.4, -2.2, 0.0, 2.0], "expected_label": None},
            {"name": "Class 1 prototype", "features": [1.2, 0.8, -0.8, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="separability_lens_lab",
        name="Separability lens lab",
        description="Strong, weak, redundant, and shortcut-like features for single-feature separation audits.",
        default_samples=220,
        input_dim=5,
        recommended_feature_map="linear",
        feature_names=("strong_signal", "weak_signal", "overlap_noise", "shortcut_code", "redundant_signal"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Clear negative", "features": [-1.0, -0.3, 0.0, -1.0, -1.0], "expected_label": 0},
            {"name": "Shortcut conflict", "features": [1.0, 0.3, 0.0, -1.0, 1.0], "expected_label": None},
            {"name": "Clear positive", "features": [1.0, 0.3, 0.0, 1.0, 1.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="neighborhood_hardness_lab",
        name="Neighborhood hardness lab",
        description="Easy cores, ambiguous boundary rows, and mislabeled islands for local hardness scans.",
        default_samples=220,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("local_signal", "support_signal", "ambiguity_band", "island_marker"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Locally easy negative", "features": [-1.2, -0.6, -0.8, 0.0], "expected_label": 0},
            {"name": "Ambiguous boundary", "features": [0.0, 0.0, 1.0, 0.0], "expected_label": None},
            {"name": "Hard island review", "features": [1.2, 0.6, -0.8, 1.6], "expected_label": None},
            {"name": "Locally easy positive", "features": [1.2, 0.6, -0.8, 0.0], "expected_label": 1},
        ),
    ),
    PresetInfo(
        key="proxy_leakage_lab",
        name="Proxy leakage lab",
        description="A label-correlated proxy feature that makes ablation and reliance diagnostics visible.",
        default_samples=180,
        input_dim=4,
        recommended_feature_map="linear",
        feature_names=("real_signal", "weak_signal", "proxy_code", "background_noise"),
        training_defaults={"epochs": 70, "batch_size": 16, "trials": 12, "feature_map": "linear"},
        prediction_examples=(
            {"name": "Signal negative", "features": [-0.8, -0.2, -1.3, 0.0], "expected_label": 0},
            {"name": "Signal positive", "features": [0.8, 0.2, 1.3, 0.0], "expected_label": 1},
            {"name": "Proxy conflict", "features": [0.8, 0.2, -1.3, 0.0], "expected_label": None},
        ),
    ),
)


def preset_labels() -> list[str]:
    return [preset.name for preset in BUILT_IN_PRESETS]


def preset_by_name(name: str) -> PresetInfo:
    for preset in BUILT_IN_PRESETS:
        if preset.name == name or preset.key == name:
            return preset
    raise ValueError(f"Unknown preset: {name}")


def preset_metadata(name: str) -> dict[str, object]:
    preset = preset_by_name(name)
    return _metadata_from_preset(preset)


def generate_builtin_preset(name: str, *, sample_count: int | None = None, seed: int = 42) -> Dataset:
    preset = preset_by_name(name)
    total = int(sample_count or preset.default_samples)
    if total < preset.min_samples:
        raise ValueError(f"{preset.name} needs at least {preset.min_samples} samples.")

    rng = np.random.default_rng(seed)
    if preset.key == "linear_blobs":
        features, labels = _linear_blobs(total, rng)
    elif preset.key == "xor":
        features, labels = _xor(total, rng)
    elif preset.key == "imbalanced_blobs":
        features, labels = _imbalanced_blobs(total, rng)
    elif preset.key == "signal_plus_noise":
        features, labels = _signal_plus_noise(total, rng)
    elif preset.key == "concentric_rings":
        features, labels = _concentric_rings(total, rng)
    elif preset.key == "two_moons":
        features, labels = _two_moons(total, rng)
    elif preset.key == "rare_event_signal":
        features, labels = _rare_event_signal(total, rng)
    elif preset.key == "overlapping_margin":
        features, labels = _overlapping_margin(total, rng)
    elif preset.key == "noisy_labels":
        features, labels = _noisy_labels(total, rng)
    elif preset.key == "sparse_interaction_signal":
        features, labels = _sparse_interaction_signal(total, rng)
    elif preset.key == "deployment_drift_probe":
        features, labels = _deployment_drift_probe(total, rng)
    elif preset.key == "active_learning_margin":
        features, labels = _active_learning_margin(total, rng)
    elif preset.key == "spurious_shortcut":
        features, labels = _spurious_shortcut(total, rng)
    elif preset.key == "subgroup_blind_spot":
        features, labels = _subgroup_blind_spot(total, rng)
    elif preset.key == "subgroup_disparity_lab":
        features, labels = _subgroup_disparity_lab(total, rng)
    elif preset.key == "response_curve_lab":
        features, labels = _response_curve_lab(total, rng)
    elif preset.key == "interaction_surface_lab":
        features, labels = _interaction_surface_lab(total, rng)
    elif preset.key == "calibration_repair_lab":
        features, labels = _calibration_repair_lab(total, rng)
    elif preset.key == "permutation_null_lab":
        features, labels = _permutation_null_lab(total, rng)
    elif preset.key == "population_drift_lab":
        features, labels = _population_drift_lab(total, rng)
    elif preset.key == "adversarial_validation_lab":
        features, labels = _adversarial_validation_lab(total, rng)
    elif preset.key == "chronological_holdout_lab":
        features, labels = _chronological_holdout_lab(total, rng)
    elif preset.key == "cost_sensitive_screening":
        features, labels = _cost_sensitive_screening(total, rng)
    elif preset.key == "decision_utility_tradeoff":
        features, labels = _decision_utility_tradeoff(total, rng)
    elif preset.key == "selective_abstention_triage":
        features, labels = _selective_abstention_triage(total, rng)
    elif preset.key == "conformal_coverage_lab":
        features, labels = _conformal_coverage_lab(total, rng)
    elif preset.key == "label_audit_traps":
        features, labels = _label_audit_traps(total, rng)
    elif preset.key == "ood_sentinel_lab":
        features, labels = _ood_sentinel_lab(total, rng)
    elif preset.key == "bootstrap_stability_lab":
        features, labels = _bootstrap_stability_lab(total, rng)
    elif preset.key == "prototype_coverage_lab":
        features, labels = _prototype_coverage_lab(total, rng)
    elif preset.key == "separability_lens_lab":
        features, labels = _separability_lens_lab(total, rng)
    elif preset.key == "neighborhood_hardness_lab":
        features, labels = _neighborhood_hardness_lab(total, rng)
    elif preset.key == "proxy_leakage_lab":
        features, labels = _proxy_leakage_lab(total, rng)
    else:
        raise ValueError(f"Unsupported preset: {preset.key}")
    return validate_dataset(features.tolist(), labels.astype(int).tolist(), min_samples=preset.min_samples, require_two_classes=True)


def save_preset_file(
    path: str | Path,
    dataset: Dataset,
    *,
    name: str,
    description: str = "",
    training_defaults: dict[str, object] | None = None,
    recommended_feature_map: str | None = None,
    feature_names: list[str] | tuple[str, ...] | None = None,
    label_names: dict[str, str] | None = None,
    prediction_examples: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
) -> Path:
    if not name or not name.strip():
        raise DataValidationError("Preset name is required.")
    output_path = Path(path)
    resolved_defaults = _sanitize_training_defaults(training_defaults)
    resolved_map = _sanitize_feature_map(recommended_feature_map or resolved_defaults.get("feature_map") or "linear")
    payload = {
        "kind": "italtensor.dataset_preset",
        "schema_version": SCHEMA_VERSION,
        "name": name.strip(),
        "description": description.strip(),
        "training_defaults": resolved_defaults,
        "recommended_feature_map": resolved_map,
        "feature_names": _sanitize_feature_names(feature_names, dataset.input_dim),
        "label_names": _sanitize_label_names(label_names),
        "prediction_examples": _sanitize_prediction_examples(prediction_examples, dataset.input_dim),
        "dataset": dataset_to_jsonable(dataset),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_preset_file(path: str | Path) -> tuple[Dataset, dict[str, Any]]:
    preset_path = Path(path)
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError("Preset file must contain a JSON object.")

    if "dataset" in payload:
        schema_version = payload.get("schema_version", payload.get("version"))
        if schema_version != SCHEMA_VERSION:
            raise DataValidationError(f"Unsupported preset schema_version: {schema_version}.")
        dataset_payload = payload["dataset"]
        metadata = {
            "name": payload.get("name", preset_path.stem),
            "description": payload.get("description", ""),
            "kind": payload.get("kind", ""),
            "schema_version": schema_version,
            "training_defaults": payload.get("training_defaults", DEFAULT_TRAINING_DEFAULTS),
            "recommended_feature_map": payload.get("recommended_feature_map"),
            "feature_names": payload.get("feature_names"),
            "label_names": payload.get("label_names"),
            "prediction_examples": payload.get("prediction_examples", []),
        }
    else:
        if "samples" not in payload:
            raise DataValidationError("Preset file must contain a dataset.")
        dataset_payload = payload
        metadata = {
            "name": preset_path.stem,
            "description": "",
            "kind": "italtensor.dataset",
            "schema_version": SCHEMA_VERSION,
            "training_defaults": DEFAULT_TRAINING_DEFAULTS,
            "recommended_feature_map": None,
            "feature_names": None,
            "label_names": {"0": "negative", "1": "positive"},
            "prediction_examples": [],
        }

    dataset = dataset_from_jsonable(dataset_payload)
    return dataset, metadata


def _sanitize_training_defaults(defaults: dict[str, object] | None) -> dict[str, object]:
    if defaults is None:
        return DEFAULT_TRAINING_DEFAULTS.copy()
    if not isinstance(defaults, dict):
        raise DataValidationError("Preset training_defaults must be a JSON object.")
    sanitized = DEFAULT_TRAINING_DEFAULTS.copy()
    for key, value in defaults.items():
        if value is None or value == "":
            continue
        if key in {"epochs", "batch_size", "trials", "feature_selection_k", "mps_bond_dim", "mps_physical_dim"}:
            parsed = int(value)
            if parsed <= 0:
                raise DataValidationError(f"Preset training default {key} must be positive.")
            sanitized[key] = parsed
        elif key in {"l1_penalty", "gradient_clip"}:
            parsed_float = float(value)
            if parsed_float < 0.0 or not np.isfinite(parsed_float):
                raise DataValidationError(f"Preset training default {key} must be a finite non-negative number.")
            sanitized[key] = parsed_float
        elif key == "feature_map":
            sanitized[key] = _sanitize_feature_map(str(value))
        elif key in {"backend", "lr_schedule"}:
            sanitized[key] = str(value).strip()
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
    return sanitized


def _sanitize_feature_map(value: str | object) -> str:
    parsed = str(value).strip().lower()
    if parsed not in {"linear", "quadratic", "rff"}:
        raise DataValidationError("Preset recommended_feature_map must be linear, quadratic, or rff.")
    return parsed


def _sanitize_feature_names(names: list[str] | tuple[str, ...] | None, input_dim: int) -> list[str]:
    if names is None:
        return [f"x{index + 1}" for index in range(input_dim)]
    if len(names) != input_dim:
        raise DataValidationError(f"Preset feature_names must contain {input_dim} name(s).")
    cleaned = [str(name).strip() or f"x{index + 1}" for index, name in enumerate(names)]
    return cleaned


def _sanitize_label_names(label_names: dict[str, str] | None) -> dict[str, str]:
    if label_names is None:
        return {"0": "negative", "1": "positive"}
    if not isinstance(label_names, dict):
        raise DataValidationError("Preset label_names must be a JSON object.")
    negative = str(label_names.get("0", label_names.get(0, "negative"))).strip() or "negative"
    positive = str(label_names.get("1", label_names.get(1, "positive"))).strip() or "positive"
    return {"0": negative, "1": positive}


def _sanitize_prediction_examples(
    examples: list[dict[str, object]] | tuple[dict[str, object], ...] | None,
    input_dim: int,
) -> list[dict[str, object]]:
    if examples is None:
        return []
    sanitized: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise DataValidationError("Preset prediction examples must be JSON objects.")
        try:
            features = np.asarray(example.get("features"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("Preset prediction example features must be numeric.") from exc
        if features.shape[0] != input_dim:
            raise DataValidationError(f"Preset prediction example {index + 1} must contain {input_dim} feature(s).")
        if not np.all(np.isfinite(features)):
            raise DataValidationError("Preset prediction example features must be finite numbers.")
        expected_label = example.get("expected_label")
        if expected_label is not None:
            try:
                parsed_label = int(expected_label)
            except (TypeError, ValueError) as exc:
                raise DataValidationError("Preset prediction example expected_label must be 0, 1, or null.") from exc
            if parsed_label not in {0, 1}:
                raise DataValidationError("Preset prediction example expected_label must be 0, 1, or null.")
            expected_label = parsed_label
        sanitized.append(
            {
                "name": str(example.get("name") or f"Example {index + 1}").strip(),
                "features": [float(value) for value in features],
                "expected_label": expected_label,
            }
        )
    return sanitized


def _linear_blobs(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    features = np.empty((total, 2), dtype=np.float32)
    features[labels == 0] = rng.normal(loc=(-1.25, -1.0), scale=0.45, size=(int(np.sum(labels == 0)), 2))
    features[labels == 1] = rng.normal(loc=(1.25, 1.0), scale=0.45, size=(int(np.sum(labels == 1)), 2))
    return _shuffle(features, labels, rng)


def _xor(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    corner_labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
    choices = np.arange(total) % 4
    rng.shuffle(choices)
    features = corners[choices] + rng.normal(0.0, 0.18, size=(total, 2)).astype(np.float32)
    labels = corner_labels[choices]
    return features, labels


def _imbalanced_blobs(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(2, round(total * 0.15))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    features = np.empty((total, 2), dtype=np.float32)
    features[:negative_count] = rng.normal(loc=(-0.35, -0.1), scale=0.65, size=(negative_count, 2))
    features[negative_count:] = rng.normal(loc=(1.15, 1.0), scale=0.35, size=(positive_count, 2))
    return _shuffle(features, labels, rng)


def _signal_plus_noise(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    informative = rng.normal(0.0, 1.0, size=(total, 2))
    score = informative[:, 0] * 1.4 - informative[:, 1] * 0.9 + rng.normal(0.0, 0.25, size=total)
    labels = (score > np.median(score)).astype(np.int32)
    noise = rng.normal(0.0, 1.0, size=(total, 4))
    features = np.concatenate([informative, noise], axis=1).astype(np.float32)
    return features, labels


def _concentric_rings(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    inner_count = int(np.sum(labels == 0))
    outer_count = int(np.sum(labels == 1))
    inner_angles = rng.uniform(0.0, 2.0 * np.pi, size=inner_count)
    outer_angles = rng.uniform(0.0, 2.0 * np.pi, size=outer_count)
    inner_radius = rng.normal(0.65, 0.07, size=inner_count)
    outer_radius = rng.normal(1.35, 0.08, size=outer_count)
    inner = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    features = np.vstack([inner, outer]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _two_moons(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    first_count = int(np.sum(labels == 0))
    second_count = int(np.sum(labels == 1))
    first_theta = rng.uniform(0.0, np.pi, size=first_count)
    second_theta = rng.uniform(0.0, np.pi, size=second_count)
    first = np.column_stack([np.cos(first_theta), np.sin(first_theta)])
    second = np.column_stack([1.0 - np.cos(second_theta), 0.45 - np.sin(second_theta)])
    features = np.vstack([first, second]) + rng.normal(0.0, 0.08, size=(total, 2))
    return _shuffle(features.astype(np.float32), labels, rng)


def _rare_event_signal(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(2, round(total * 0.08))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    negatives = rng.normal(0.0, 0.85, size=(negative_count, 4))
    positives = rng.normal((1.7, 1.5, 0.0, 0.0), (0.25, 0.25, 1.0, 1.0), size=(positive_count, 4))
    features = np.vstack([negatives, positives]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _overlapping_margin(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    covariance = np.asarray([[0.42, 0.24], [0.24, 0.38]], dtype=np.float32)
    negatives = rng.multivariate_normal(mean=(-0.55, -0.35), cov=covariance, size=negative_count)
    positives = rng.multivariate_normal(mean=(0.55, 0.35), cov=covariance, size=positive_count)
    margin_count = max(2, total // 12)
    margin_indices = rng.choice(total, size=margin_count, replace=False)
    features = np.vstack([negatives, positives]).astype(np.float32)
    features[margin_indices] = rng.normal(0.0, 0.18, size=(margin_count, 2)).astype(np.float32)
    return _shuffle(features, labels, rng)


def _noisy_labels(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _linear_blobs(total, rng)
    flip_count = max(2, int(round(total * 0.12)))
    flip_indices = rng.choice(total, size=flip_count, replace=False)
    labels = labels.copy()
    labels[flip_indices] = 1 - labels[flip_indices]
    return features.astype(np.float32), labels.astype(np.int32)


def _sparse_interaction_signal(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features = rng.normal(0.0, 1.0, size=(total, 16)).astype(np.float32)
    score = (
        1.2 * features[:, 0]
        + 0.9 * features[:, 3]
        - 1.0 * features[:, 7]
        + 1.4 * features[:, 0] * features[:, 1]
        + rng.normal(0.0, 0.35, size=total)
    )
    labels = (score > np.median(score)).astype(np.int32)
    features[:, 2] = features[:, 0] * 0.75 + rng.normal(0.0, 0.2, size=total)
    return features.astype(np.float32), labels


def _deployment_drift_probe(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    features = np.empty((total, 4), dtype=np.float32)
    features[labels == 0, :2] = rng.normal(loc=(-0.85, -0.65), scale=0.28, size=(negative_count, 2))
    features[labels == 1, :2] = rng.normal(loc=(0.85, 0.65), scale=0.28, size=(positive_count, 2))
    features[:, 2:] = rng.normal(0.0, 0.35, size=(total, 2))
    return _shuffle(features, labels, rng)


def _active_learning_margin(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    margin_count = max(4, total // 5)
    core_negative = max(2, negative_count - margin_count // 2)
    core_positive = max(2, positive_count - (margin_count - margin_count // 2))
    negative_margin = negative_count - core_negative
    positive_margin = positive_count - core_positive

    negatives = rng.normal(loc=(-1.1, -0.45), scale=(0.32, 0.28), size=(core_negative, 2))
    positives = rng.normal(loc=(1.1, 0.45), scale=(0.32, 0.28), size=(core_positive, 2))
    margin_negatives = rng.normal(loc=(-0.12, 0.0), scale=(0.18, 0.32), size=(negative_margin, 2))
    margin_positives = rng.normal(loc=(0.12, 0.0), scale=(0.18, 0.32), size=(positive_margin, 2))
    features = np.vstack([negatives, margin_negatives, positives, margin_positives]).astype(np.float32)
    labels = np.asarray(
        [0] * (core_negative + negative_margin) + [1] * (core_positive + positive_margin),
        dtype=np.int32,
    )
    return _shuffle(features, labels, rng)


def _spurious_shortcut(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    signed = np.where(labels == 1, 1.0, -1.0)
    stable_signal = signed * 0.55 + rng.normal(0.0, 0.55, size=total)
    context_noise = rng.normal(0.0, 1.0, size=total)
    shortcut_signal = signed * 1.2 + rng.normal(0.0, 0.12, size=total)
    conflict_count = max(2, total // 12)
    conflict_indices = rng.choice(total, size=conflict_count, replace=False)
    stable_signal[conflict_indices] *= -1.0
    features = np.column_stack([stable_signal, context_noise, shortcut_signal]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _subgroup_blind_spot(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    subgroup = rng.random(total) < 0.22
    primary_signal = rng.normal(0.0, 1.0, size=total)
    context_noise = rng.normal(0.0, 1.0, size=total)
    majority_score = primary_signal + rng.normal(0.0, 0.25, size=total)
    minority_score = -primary_signal + 0.25 * context_noise + rng.normal(0.0, 0.25, size=total)
    score = np.where(subgroup, minority_score, majority_score)
    labels = (score > 0.0).astype(np.int32)
    subgroup_marker = subgroup.astype(np.float32) + rng.normal(0.0, 0.04, size=total)
    features = np.column_stack([primary_signal, subgroup_marker, context_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _subgroup_disparity_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    subgroup = rng.random(total) < 0.28
    primary_signal = rng.normal(0.0, 1.0, size=total)
    context_noise = rng.normal(0.0, 1.0, size=total)
    majority_score = primary_signal + 0.35 * context_noise + rng.normal(0.0, 0.35, size=total)
    minority_score = -0.85 * primary_signal + 0.25 * context_noise + rng.normal(0.0, 0.55, size=total)
    score = np.where(subgroup, minority_score, majority_score)
    labels = (score > np.median(score)).astype(np.int32)
    features = np.column_stack([primary_signal, subgroup.astype(np.float32), context_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _response_curve_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    linear_driver = rng.normal(0.0, 1.0, size=total)
    nonlinear_arc = rng.uniform(-1.8, 1.8, size=total)
    weak_noise = rng.normal(0.0, 1.0, size=total)
    arc_signal = 1.25 - 1.15 * np.square(nonlinear_arc)
    latent = 0.85 * linear_driver + arc_signal + 0.12 * weak_noise + rng.normal(0.0, 0.35, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    features = np.column_stack([linear_driver, nonlinear_arc, weak_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _interaction_surface_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left_factor = rng.normal(0.0, 1.0, size=total)
    right_factor = rng.normal(0.0, 1.0, size=total)
    weak_signal = rng.normal(0.0, 1.0, size=total)
    distractor_noise = rng.normal(0.0, 1.0, size=total)
    latent = 1.4 * left_factor * right_factor + 0.3 * weak_signal + rng.normal(0.0, 0.35, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    features = np.column_stack([left_factor, right_factor, weak_signal, distractor_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _calibration_repair_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    margin_score = rng.normal(0.0, 1.0, size=total)
    confidence_trap = np.abs(margin_score) + rng.normal(0.0, 0.2, size=total)
    background_noise = rng.normal(0.0, 1.0, size=total)
    shoulder = np.abs(margin_score) < 0.75
    base_probability = 1.0 / (1.0 + np.exp(-(1.35 * margin_score + 0.15 * background_noise)))
    noisy_probability = np.where(shoulder, 0.35 + 0.3 * base_probability, base_probability)
    labels = (rng.random(total) < noisy_probability).astype(np.int32)
    features = np.column_stack([margin_score, confidence_trap, background_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _permutation_null_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    real_margin = rng.normal(0.0, 1.0, size=total)
    weak_support = rng.normal(0.0, 1.0, size=total)
    decoy_noise = rng.normal(0.0, 1.0, size=total)
    latent = 1.25 * real_margin + 0.45 * weak_support + rng.normal(0.0, 0.45, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    label_noise_marker = np.abs(latent - np.median(latent)) < 0.35
    flip_candidates = np.where(label_noise_marker)[0]
    if flip_candidates.size:
        flip_count = min(flip_candidates.size, max(1, total // 12))
        flip_indices = rng.choice(flip_candidates, size=flip_count, replace=False)
        labels[flip_indices] = 1 - labels[flip_indices]
    marker_feature = label_noise_marker.astype(np.float32) + rng.normal(0.0, 0.08, size=total)
    features = np.column_stack([real_margin, weak_support, marker_feature, decoy_noise]).astype(np.float32)
    return _shuffle(features, labels.astype(np.int32), rng)


def _population_drift_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    reference_count = total // 2
    current_count = total - reference_count
    stable_reference = rng.normal(0.0, 1.0, size=reference_count)
    stable_current = rng.normal(0.0, 1.0, size=current_count)
    shifted_reference = rng.normal(0.0, 0.35, size=reference_count)
    shifted_current = rng.normal(1.35, 0.35, size=current_count)
    variance_reference = rng.normal(0.0, 0.35, size=reference_count)
    variance_current = rng.normal(0.0, 1.1, size=current_count)
    tail_reference = rng.normal(0.0, 0.4, size=reference_count)
    tail_current = rng.normal(0.0, 0.4, size=current_count)
    tail_count = max(2, current_count // 10)
    tail_current[:tail_count] = rng.normal(3.0, 0.25, size=tail_count)
    decoy_reference = rng.normal(0.0, 1.0, size=reference_count)
    decoy_current = rng.normal(0.0, 1.0, size=current_count)

    reference = np.column_stack(
        [stable_reference, shifted_reference, variance_reference, tail_reference, decoy_reference]
    )
    current = np.column_stack(
        [stable_current, shifted_current, variance_current, tail_current, decoy_current]
    )
    features = np.vstack([reference, current]).astype(np.float32)
    latent = 1.0 * features[:, 0] + 0.25 * features[:, 1] + rng.normal(0.0, 0.45, size=total)
    labels = (latent > np.quantile(latent, 0.52)).astype(np.int32)
    return features, labels.astype(np.int32)


def _adversarial_validation_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    reference_count = total // 2
    current_count = total - reference_count
    stable_reference = rng.normal(0.0, 1.0, size=reference_count)
    stable_current = rng.normal(0.0, 1.0, size=current_count)
    axis_reference = rng.normal(0.0, 1.0, size=reference_count)
    axis_current = rng.normal(0.25, 1.0, size=current_count)
    interaction_reference = axis_reference + rng.normal(0.0, 0.18, size=reference_count)
    interaction_current = -axis_current + rng.normal(0.0, 0.18, size=current_count)
    variance_reference = rng.normal(0.0, 0.45, size=reference_count)
    variance_current = rng.normal(0.0, 1.25, size=current_count)
    decoy_reference = rng.normal(0.0, 1.0, size=reference_count)
    decoy_current = rng.normal(0.0, 1.0, size=current_count)
    reference = np.column_stack(
        [stable_reference, axis_reference, interaction_reference, variance_reference, decoy_reference]
    )
    current = np.column_stack(
        [stable_current, axis_current, interaction_current, variance_current, decoy_current]
    )
    features = np.vstack([reference, current]).astype(np.float32)
    latent = 1.0 * features[:, 0] + 0.25 * features[:, 1] + rng.normal(0.0, 0.45, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    return features, labels.astype(np.int32)


def _chronological_holdout_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    reference_count = max(8, int(round(total * 0.6)))
    reference_count = min(reference_count, total - 4)
    current_count = total - reference_count

    reference_early_signal = rng.normal(0.0, 1.0, size=reference_count)
    reference_late_signal = reference_early_signal * 0.65 + rng.normal(0.0, 0.45, size=reference_count)
    reference_threshold = rng.normal(0.0, 0.35, size=reference_count)
    reference_wobble = rng.normal(0.0, 0.25, size=reference_count)
    reference_decoy = rng.normal(0.0, 1.0, size=reference_count)
    reference_latent = (
        1.05 * reference_early_signal
        + 0.75 * reference_late_signal
        - 0.25 * reference_threshold
        + rng.normal(0.0, 0.35, size=reference_count)
    )
    reference_labels = (reference_latent > np.median(reference_latent)).astype(np.int32)

    current_early_signal = rng.normal(0.0, 1.0, size=current_count)
    current_late_signal = current_early_signal * 0.15 + rng.normal(0.0, 0.9, size=current_count)
    current_threshold = rng.normal(1.0, 0.45, size=current_count)
    current_wobble = rng.normal(0.5, 0.65, size=current_count)
    current_decoy = rng.normal(0.0, 1.0, size=current_count)
    current_latent = (
        0.20 * current_early_signal
        - 1.05 * current_late_signal
        + 0.65 * current_threshold
        + 0.25 * current_wobble
        + rng.normal(0.0, 0.45, size=current_count)
    )
    current_labels = (current_latent > np.median(current_latent)).astype(np.int32)

    reference = np.column_stack(
        [
            reference_early_signal,
            reference_late_signal,
            reference_threshold,
            reference_wobble,
            reference_decoy,
        ]
    )
    current = np.column_stack(
        [
            current_early_signal,
            current_late_signal,
            current_threshold,
            current_wobble,
            current_decoy,
        ]
    )
    features = np.vstack([reference, current]).astype(np.float32)
    labels = np.concatenate([reference_labels, current_labels]).astype(np.int32)
    return features, labels


def _cost_sensitive_screening(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(3, round(total * 0.12))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    negatives = rng.normal(loc=(-0.25, -0.15, 0.0), scale=(0.55, 0.45, 1.0), size=(negative_count, 3))
    positives = rng.normal(loc=(0.65, 0.45, 0.0), scale=(0.45, 0.4, 1.0), size=(positive_count, 3))
    features = np.vstack([negatives, positives]).astype(np.float32)
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    return _shuffle(features, labels, rng)


def _decision_utility_tradeoff(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positive_count = max(4, round(total * 0.18))
    negative_count = total - positive_count
    if negative_count < 2:
        negative_count = 2
        positive_count = total - negative_count
    negatives = rng.normal(loc=(-0.25, -0.15, 0.0, -0.15), scale=(0.5, 0.45, 1.0, 0.55), size=(negative_count, 4))
    positives = rng.normal(loc=(0.65, 0.45, 0.0, 0.35), scale=(0.45, 0.4, 1.0, 0.5), size=(positive_count, 4))
    hard_negative_count = max(2, negative_count // 8)
    hard_positive_count = max(2, positive_count // 4)
    negatives[:hard_negative_count, :2] = rng.normal(loc=(0.35, 0.25), scale=(0.25, 0.25), size=(hard_negative_count, 2))
    positives[:hard_positive_count, :2] = rng.normal(loc=(0.15, 0.1), scale=(0.25, 0.25), size=(hard_positive_count, 2))
    features = np.vstack([negatives, positives]).astype(np.float32)
    labels = np.asarray([0] * negative_count + [1] * positive_count, dtype=np.int32)
    return _shuffle(features, labels, rng)


def _selective_abstention_triage(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    ambiguous_count = max(6, total // 4)
    ambiguous_negative = ambiguous_count // 2
    ambiguous_positive = ambiguous_count - ambiguous_negative
    core_negative = max(2, negative_count - ambiguous_negative)
    core_positive = max(2, positive_count - ambiguous_positive)
    negative_boundary = negative_count - core_negative
    positive_boundary = positive_count - core_positive

    negatives = rng.normal(loc=(-1.1, -0.6, -0.8), scale=(0.28, 0.25, 0.25), size=(core_negative, 3))
    positives = rng.normal(loc=(1.1, 0.6, -0.8), scale=(0.28, 0.25, 0.25), size=(core_positive, 3))
    boundary_negatives = rng.normal(loc=(-0.08, -0.03, 1.0), scale=(0.18, 0.25, 0.25), size=(negative_boundary, 3))
    boundary_positives = rng.normal(loc=(0.08, 0.03, 1.0), scale=(0.18, 0.25, 0.25), size=(positive_boundary, 3))
    features = np.vstack([negatives, boundary_negatives, positives, boundary_positives]).astype(np.float32)
    labels = np.asarray(
        [0] * (core_negative + negative_boundary) + [1] * (core_positive + positive_boundary),
        dtype=np.int32,
    )
    return _shuffle(features, labels, rng)


def _conformal_coverage_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    shoulder_count = max(8, total // 3)
    shoulder_negative = min(negative_count - 2, shoulder_count // 2)
    shoulder_positive = min(positive_count - 2, shoulder_count - shoulder_negative)
    core_negative = negative_count - shoulder_negative
    core_positive = positive_count - shoulder_positive

    negative_core = rng.normal(loc=(-1.25, -0.35, -0.8), scale=(0.24, 0.45, 0.25), size=(core_negative, 3))
    positive_core = rng.normal(loc=(1.25, 0.35, -0.8), scale=(0.24, 0.45, 0.25), size=(core_positive, 3))
    negative_shoulder = rng.normal(loc=(-0.18, 0.05, 1.0), scale=(0.28, 0.55, 0.18), size=(shoulder_negative, 3))
    positive_shoulder = rng.normal(loc=(0.18, -0.05, 1.0), scale=(0.28, 0.55, 0.18), size=(shoulder_positive, 3))
    features = np.vstack([negative_core, negative_shoulder, positive_core, positive_shoulder]).astype(np.float32)
    output_labels = np.asarray(
        [0] * (core_negative + shoulder_negative) + [1] * (core_positive + shoulder_positive),
        dtype=np.int32,
    )
    return _shuffle(features, output_labels, rng)


def _label_audit_traps(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _linear_blobs(total, rng)
    flip_count = max(2, int(round(total * 0.08)))
    negative_indices = np.where(labels == 0)[0]
    positive_indices = np.where(labels == 1)[0]
    chosen_negatives = rng.choice(negative_indices, size=flip_count // 2, replace=False)
    chosen_positives = rng.choice(positive_indices, size=flip_count - chosen_negatives.shape[0], replace=False)
    labels = labels.copy()
    labels[chosen_negatives] = 1
    labels[chosen_positives] = 0
    return features.astype(np.float32), labels.astype(np.int32)


def _ood_sentinel_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    signed = np.where(labels == 1, 1.0, -1.0)
    main_signal = signed * 0.9 + rng.normal(0.0, 0.35, size=total)
    support_signal = signed * 0.5 + rng.normal(0.0, 0.45, size=total)
    leverage_axis = rng.normal(0.0, 0.35, size=total)
    artifact_code = rng.normal(0.0, 0.25, size=total)

    outlier_count = max(4, total // 12)
    outlier_indices = rng.choice(total, size=outlier_count, replace=False)
    split = outlier_count // 2
    high = outlier_indices[:split]
    low = outlier_indices[split:]
    leverage_axis[high] = rng.normal(4.5, 0.35, size=high.shape[0])
    artifact_code[high] = rng.normal(-3.5, 0.25, size=high.shape[0])
    leverage_axis[low] = rng.normal(-4.5, 0.35, size=low.shape[0])
    artifact_code[low] = rng.normal(3.5, 0.25, size=low.shape[0])

    conflict_count = max(2, outlier_count // 3)
    conflict_indices = outlier_indices[:conflict_count]
    labels = labels.copy()
    labels[conflict_indices] = 1 - labels[conflict_indices]

    features = np.column_stack([main_signal, support_signal, leverage_axis, artifact_code]).astype(np.float32)
    return _shuffle(features, labels.astype(np.int32), rng)


def _bootstrap_stability_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    boundary_count = max(12, total // 4)
    boundary_negative = min(negative_count - 2, boundary_count // 2)
    boundary_positive = min(positive_count - 2, boundary_count - boundary_negative)
    core_negative = negative_count - boundary_negative
    core_positive = positive_count - boundary_positive

    negative_core = rng.normal(loc=(-1.15, -0.45, -0.8, 0.0), scale=(0.25, 0.28, 0.25, 0.8), size=(core_negative, 4))
    positive_core = rng.normal(loc=(1.15, 0.45, -0.8, 0.0), scale=(0.25, 0.28, 0.25, 0.8), size=(core_positive, 4))
    negative_boundary = rng.normal(loc=(-0.08, 0.05, 1.1, 0.0), scale=(0.28, 0.45, 0.22, 0.8), size=(boundary_negative, 4))
    positive_boundary = rng.normal(loc=(0.08, -0.05, 1.1, 0.0), scale=(0.28, 0.45, 0.22, 0.8), size=(boundary_positive, 4))
    features = np.vstack([negative_core, negative_boundary, positive_core, positive_boundary]).astype(np.float32)
    output_labels = np.asarray(
        [0] * (core_negative + boundary_negative) + [1] * (core_positive + boundary_positive),
        dtype=np.int32,
    )

    flip_candidates = np.where(features[:, 2] > 0.8)[0]
    if flip_candidates.size:
        flip_count = min(flip_candidates.size, max(2, total // 20))
        flip_indices = rng.choice(flip_candidates, size=flip_count, replace=False)
        output_labels[flip_indices] = 1 - output_labels[flip_indices]
    return _shuffle(features, output_labels, rng)


def _prototype_coverage_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    boundary_count = max(12, total // 5)
    island_count = max(6, total // 10)
    core_count = max(4, total - boundary_count - island_count)
    negative_core_count = core_count // 2
    positive_core_count = core_count - negative_core_count
    negative_boundary_count = boundary_count // 2
    positive_boundary_count = boundary_count - negative_boundary_count
    negative_island_count = island_count // 2
    positive_island_count = island_count - negative_island_count

    negative_core = rng.normal(
        loc=(-1.2, -0.75, -0.8, 0.0),
        scale=(0.22, 0.24, 0.18, 0.18),
        size=(negative_core_count, 4),
    )
    positive_core = rng.normal(
        loc=(1.2, 0.75, -0.8, 0.0),
        scale=(0.22, 0.24, 0.18, 0.18),
        size=(positive_core_count, 4),
    )
    negative_boundary = rng.normal(
        loc=(-0.08, 0.05, 1.0, 0.0),
        scale=(0.20, 0.28, 0.12, 0.18),
        size=(negative_boundary_count, 4),
    )
    positive_boundary = rng.normal(
        loc=(0.08, -0.05, 1.0, 0.0),
        scale=(0.20, 0.28, 0.12, 0.18),
        size=(positive_boundary_count, 4),
    )
    negative_island = rng.normal(
        loc=(2.4, -2.2, 0.0, 2.0),
        scale=(0.18, 0.20, 0.18, 0.16),
        size=(negative_island_count, 4),
    )
    positive_island = rng.normal(
        loc=(-2.4, 2.2, 0.0, -2.0),
        scale=(0.18, 0.20, 0.18, 0.16),
        size=(positive_island_count, 4),
    )
    features = np.vstack(
        [
            negative_core,
            negative_boundary,
            negative_island,
            positive_core,
            positive_boundary,
            positive_island,
        ]
    ).astype(np.float32)
    labels = np.asarray(
        [0] * (negative_core_count + negative_boundary_count + negative_island_count)
        + [1] * (positive_core_count + positive_boundary_count + positive_island_count),
        dtype=np.int32,
    )
    return _shuffle(features, labels, rng)


def _separability_lens_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    strong_signal = rng.normal(0.0, 1.0, size=total)
    weak_signal = rng.normal(0.0, 1.0, size=total)
    latent = 1.35 * strong_signal + 0.35 * weak_signal + rng.normal(0.0, 0.45, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    signed = np.where(labels == 1, 1.0, -1.0)
    overlap_noise = rng.normal(0.0, 1.0, size=total)
    shortcut_code = signed + rng.normal(0.0, 0.025, size=total)
    conflict_count = max(2, total // 20)
    conflict_indices = rng.choice(total, size=conflict_count, replace=False)
    shortcut_code[conflict_indices] *= -1.0
    redundant_signal = strong_signal + rng.normal(0.0, 0.05, size=total)
    features = np.column_stack(
        [strong_signal, weak_signal, overlap_noise, shortcut_code, redundant_signal]
    ).astype(np.float32)
    return _shuffle(features, labels, rng)


def _neighborhood_hardness_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(total)
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    boundary_count = max(12, total // 5)
    island_count = max(6, total // 12)
    boundary_negative = min(negative_count - 4, boundary_count // 2)
    boundary_positive = min(positive_count - 4, boundary_count - boundary_negative)
    island_negative = min(negative_count - boundary_negative - 2, island_count // 2)
    island_positive = min(positive_count - boundary_positive - 2, island_count - island_negative)
    core_negative = negative_count - boundary_negative - island_negative
    core_positive = positive_count - boundary_positive - island_positive

    negative_core = rng.normal(loc=(-1.2, -0.6, -0.8, 0.0), scale=(0.22, 0.24, 0.20, 0.20), size=(core_negative, 4))
    positive_core = rng.normal(loc=(1.2, 0.6, -0.8, 0.0), scale=(0.22, 0.24, 0.20, 0.20), size=(core_positive, 4))
    negative_boundary = rng.normal(loc=(-0.08, 0.02, 1.0, 0.0), scale=(0.25, 0.32, 0.14, 0.20), size=(boundary_negative, 4))
    positive_boundary = rng.normal(loc=(0.08, -0.02, 1.0, 0.0), scale=(0.25, 0.32, 0.14, 0.20), size=(boundary_positive, 4))
    negative_island = rng.normal(loc=(1.1, 0.6, -0.8, 1.6), scale=(0.16, 0.18, 0.14, 0.12), size=(island_negative, 4))
    positive_island = rng.normal(loc=(-1.1, -0.6, -0.8, -1.6), scale=(0.16, 0.18, 0.14, 0.12), size=(island_positive, 4))
    features = np.vstack(
        [negative_core, negative_boundary, negative_island, positive_core, positive_boundary, positive_island]
    ).astype(np.float32)
    output_labels = np.asarray(
        [0] * (core_negative + boundary_negative + island_negative)
        + [1] * (core_positive + boundary_positive + island_positive),
        dtype=np.int32,
    )
    return _shuffle(features, output_labels, rng)


def _proxy_leakage_lab(total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    real_signal = rng.normal(0.0, 1.0, size=total)
    weak_signal = rng.normal(0.0, 1.0, size=total)
    latent = 0.9 * real_signal + 0.35 * weak_signal + rng.normal(0.0, 0.35, size=total)
    labels = (latent > np.median(latent)).astype(np.int32)
    signed = np.where(labels == 1, 1.0, -1.0)
    proxy_code = signed * 1.35 + rng.normal(0.0, 0.08, size=total)
    conflict_count = max(2, total // 12)
    conflict_indices = rng.choice(total, size=conflict_count, replace=False)
    proxy_code[conflict_indices] *= -1.0
    background_noise = rng.normal(0.0, 1.0, size=total)
    features = np.column_stack([real_signal, weak_signal, proxy_code, background_noise]).astype(np.float32)
    return _shuffle(features, labels, rng)


def _balanced_labels(total: int) -> np.ndarray:
    labels = np.asarray([0] * (total // 2) + [1] * (total - total // 2), dtype=np.int32)
    return labels


def _shuffle(features: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.permutation(labels.shape[0])
    return features[indices].astype(np.float32), labels[indices].astype(np.int32)


def _metadata_from_preset(preset: PresetInfo) -> dict[str, object]:
    return {
        "name": preset.name,
        "description": preset.description,
        "schema_version": SCHEMA_VERSION,
        "input_dim": preset.input_dim,
        "recommended_feature_map": preset.recommended_feature_map,
        "feature_names": list(preset.feature_names),
        "label_names": {"0": preset.label_names[0], "1": preset.label_names[1]},
        "training_defaults": dict(preset.training_defaults),
        "prediction_examples": [dict(example) for example in preset.prediction_examples],
    }
