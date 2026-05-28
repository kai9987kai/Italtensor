from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .data import (
    DataValidationError,
    load_csv_dataset,
    parse_prediction_vector,
    parse_training_example,
    validate_dataset,
)
from .calibration_repair import format_calibration_repair_summary, run_calibration_repair_diagnostics
from .counterfactual import find_counterfactual, format_counterfactual_result
from .conformal_sets import format_conformal_set_summary, run_conformal_set_diagnostics
from .decision_curve import format_decision_curve_summary, run_decision_curve_diagnostics
from .experiments import (
    ExperimentResult,
    conformal_label_set,
    run_experiments,
    select_best_result,
    train_single_model,
    train_single_model_cv,
)
from .modeling import ModelConfig, NumpyBinaryClassifier, predict_probability
from .model_communication import FUSION_CHOICES, ModelPanel, PanelMember, fit_stacking_weights
from .model_response import format_model_response_summary, run_model_response_diagnostics
from .model_runner import ModelRunQueue, available_backends, run_model_queue, select_best_from_runs
from .pairwise_interactions import format_pairwise_interaction_summary, run_pairwise_interaction_diagnostics
from .permutation_null import format_permutation_null_summary, run_permutation_null_diagnostics
from .population_drift import format_population_drift_summary, run_population_drift_diagnostics
from .adversarial_validation import (
    format_adversarial_validation_summary,
    run_adversarial_validation_diagnostics,
)
from .chronological_holdout import (
    format_chronological_holdout_summary,
    run_chronological_holdout_diagnostics,
)
from .ablation import format_ablation_summary, run_ablation_diagnostics
from .persistence import (
    load_dataset,
    load_model_bundle,
    load_model_registry,
    save_dataset,
    save_model_bundle,
    save_model_registry,
)
from .preprocessing import FeatureStandardizer
from .presets import generate_builtin_preset, load_preset_file, preset_labels, preset_metadata, save_preset_file
from .reporting import build_experiment_report, export_experiment_report
from .registry import ModelSlot
from .sample_review import format_sample_review_summary, run_sample_review
from .scoring import load_reviewed_prediction_csv, score_prediction_csv
from .selective_risk import format_selective_risk_summary, run_selective_risk_diagnostics
from .subgroup_disparity import format_subgroup_disparity_summary, run_subgroup_disparity_diagnostics
from .audit import audit_dataset, format_audit_summary
from .learning_curves import learning_curve_points
from .slices import format_slice_summary, run_slice_diagnostics
from .stress import format_stress_summary, run_stress_suite
from .thresholds import format_threshold_summary, run_threshold_diagnostics
from .cartography import format_cartography_summary, run_dataset_cartography
from .mps_diagnostics import format_mps_sweep_summary, run_mps_bond_sweep
from .ood_sentinel import format_ood_sentinel_summary, run_ood_sentinel
from .dataset_triage import format_dataset_triage_summary, run_dataset_triage
from .experiment_advisor import build_experiment_advisor, format_experiment_advisor_summary
from .trial_inspector import inspect_trial_history, format_trial_inspector_summary
from .bootstrap_stability import (
    format_bootstrap_stability_summary,
    run_bootstrap_stability_diagnostics,
)
from .prototype_audit import format_prototype_audit_summary, run_prototype_audit
from .feature_separability import (
    format_feature_separability_summary,
    run_feature_separability_diagnostics,
)
from .neighborhood_hardness import (
    format_neighborhood_hardness_summary,
    run_neighborhood_hardness_diagnostics,
)
from .trials_io import export_trial_history_csv
from . import __version__


@dataclass
class AppState:
    features: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    input_dim: int | None = None
    model: Any | None = None
    latest_config: ModelConfig | None = None
    latest_metrics: dict[str, float | int] = field(default_factory=dict)
    latest_threshold: float = 0.5
    preprocessor: FeatureStandardizer | None = None
    feature_importances: list[dict[str, float | int]] = field(default_factory=list)
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_metadata: dict[str, Any] = field(default_factory=dict)
    latest_ablation_report: dict[str, Any] | None = None
    latest_decision_curve_report: dict[str, Any] | None = None
    latest_conformal_set_report: dict[str, Any] | None = None
    latest_selective_risk_report: dict[str, Any] | None = None
    latest_sample_review_report: dict[str, Any] | None = None
    latest_threshold_report: dict[str, Any] | None = None
    latest_calibration_repair_report: dict[str, Any] | None = None
    latest_model_response_report: dict[str, Any] | None = None
    latest_pairwise_interaction_report: dict[str, Any] | None = None
    latest_slice_report: dict[str, Any] | None = None
    latest_subgroup_disparity_report: dict[str, Any] | None = None
    latest_stress_report: dict[str, Any] | None = None
    latest_permutation_null_report: dict[str, Any] | None = None
    latest_population_drift_report: dict[str, Any] | None = None
    latest_adversarial_validation_report: dict[str, Any] | None = None
    latest_chronological_holdout_report: dict[str, Any] | None = None
    latest_cartography_report: dict[str, Any] | None = None
    latest_ood_sentinel_report: dict[str, Any] | None = None
    latest_bootstrap_stability_report: dict[str, Any] | None = None
    latest_prototype_audit_report: dict[str, Any] | None = None
    latest_feature_separability_report: dict[str, Any] | None = None
    latest_neighborhood_hardness_report: dict[str, Any] | None = None
    latest_dataset_triage_report: dict[str, Any] | None = None
    latest_experiment_advisor_report: dict[str, Any] | None = None
    latest_trial_inspector_report: dict[str, Any] | None = None
    latest_mps_sweep_report: dict[str, Any] | None = None
    busy: bool = False
    status_message: str = "Ready"
    model_slots: list[ModelSlot] = field(default_factory=list)
    active_slot_index: int | None = None
    panel_fusion: str = "mean"
    communication_log: list[dict[str, Any]] = field(default_factory=list)


def run_app() -> None:
    try:
        import PySimpleGUI as sg
    except ImportError as exc:
        raise RuntimeError(
            "PySimpleGUI is not installed. Install GUI dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    theme_dict = {
        'BACKGROUND': '#0E0F11',
        'TEXT': '#E4E6EB',
        'INPUT': '#1C1D21',
        'TEXT_INPUT': '#FFFFFF',
        'SCROLL': '#1C1D21',
        'BUTTON': ('#FFFFFF', '#6366F1'),
        'PROGRESS': ('#6366F1', '#1C1D21'),
        'BORDER': 0,
        'SLIDER_DEPTH': 0,
        'PROGRESS_DEPTH': 0,
    }
    sg.theme_add_new('ItaltensorPremiumDark', theme_dict)
    sg.theme('ItaltensorPremiumDark')
    sg.set_options(font=('Segoe UI', 10))
    
    state = AppState()
    window = sg.Window(f"Italtensor Workbench v{__version__}", _layout(sg), finalize=True, resizable=True)
    _refresh_state(window, state)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        try:
            if event == "-ADD_SAMPLE-":
                _add_sample(window, state, values)
            elif event == "-LOAD_BUILTIN_PRESET-":
                _load_builtin_preset(window, state, values)
            elif event == "-IMPORT_PRESET-":
                _import_preset(window, state, values)
            elif event == "-SAVE_PRESET-":
                _save_preset(window, state, values)
            elif event == "-LOAD_CSV-":
                _load_csv(window, state, values)
            elif event == "-SAVE_DATASET-":
                _save_dataset(window, state, values)
            elif event == "-LOAD_DATASET-":
                _load_dataset(window, state, values)
            elif event == "-CLEAR_DATA-":
                _clear_data(window, state)
            elif event == "-AUDIT_DATASET-":
                _audit_dataset(window, state)
            elif event == "-POPULATION_DRIFT-":
                _start_population_drift(window, state)
            elif event == "-ADVERSARIAL_VALIDATION-":
                _start_adversarial_validation(window, state)
            elif event == "-CHRONOLOGICAL_HOLDOUT-":
                _start_chronological_holdout(window, state)
            elif event == "-LEARNING_CURVE-":
                _start_learning_curve(window, state, values)
            elif event == "-ABLATION_DIAGNOSTICS-":
                _start_ablation_diagnostics(window, state)
            elif event == "-DECISION_CURVE-":
                _start_decision_curve(window, state)
            elif event == "-CONFORMAL_SETS-":
                _start_conformal_sets(window, state)
            elif event == "-CALIBRATION_REPAIR-":
                _start_calibration_repair(window, state)
            elif event == "-SELECTIVE_RISK-":
                _start_selective_risk(window, state)
            elif event == "-STRESS_TEST-":
                _start_stress_test(window, state)
            elif event == "-SLICE_DIAGNOSTICS-":
                _start_slice_diagnostics(window, state)
            elif event == "-MODEL_RESPONSE-":
                _start_model_response(window, state)
            elif event == "-PAIRWISE_INTERACTIONS-":
                _start_pairwise_interactions(window, state)
            elif event == "-SUBGROUP_DISPARITY-":
                _start_subgroup_disparity(window, state)
            elif event == "-THRESHOLD_DIAGNOSTICS-":
                _start_threshold_diagnostics(window, state)
            elif event == "-SAMPLE_REVIEW-":
                _start_sample_review(window, state)
            elif event == "-PERMUTATION_NULL-":
                _start_permutation_null(window, state)
            elif event == "-CARTOGRAPHY-":
                _start_cartography(window, state)
            elif event == "-DATASET_TRIAGE-":
                _start_dataset_triage(window, state)
            elif event == "-EXPERIMENT_ADVISOR-":
                _start_experiment_advisor(window, state)
            elif event == "-TRIAL_INSPECTOR-":
                _start_trial_inspector(window, state)
            elif event == "-OOD_SENTINEL-":
                _start_ood_sentinel(window, state)
            elif event == "-BOOTSTRAP_STABILITY-":
                _start_bootstrap_stability(window, state, values)
            elif event == "-PROTOTYPE_AUDIT-":
                _start_prototype_audit(window, state)
            elif event == "-FEATURE_SEPARABILITY-":
                _start_feature_separability(window, state)
            elif event == "-NEIGHBORHOOD_HARDNESS-":
                _start_neighborhood_hardness(window, state)
            elif event == "-RELIABILITY-":
                _run_reliability_diagram(window, state)
            elif event == "-MPS_BOND_SWEEP-":
                _start_mps_bond_sweep(window, state, values)
            elif event == "-SHAP_ANALYSIS-":
                _run_shap_analysis(window, state, values)
            elif event == "-DECISION_BOUNDARY-":
                _run_decision_boundary(window, state, values)
            elif event == "-EXPORT_TRIALS-":
                _export_trials(window, state, values)
            elif event == "-SLOT_SIMILARITY-":
                _run_slot_similarity(window, state)
            elif event == "-TRAIN_ONCE-":
                _start_train_once(window, state, values)
            elif event == "-AUTO_EXPERIMENTS-":
                _start_auto_experiments(window, state, values)
            elif event == "-SAVE_MODEL-":
                _save_model(window, state, values)
            elif event == "-LOAD_MODEL-":
                _load_model(window, state, values)
            elif event == "-EXPORT_REPORT-":
                _export_report(window, state, values)
            elif event == "-PREDICT-":
                _predict(window, state, values)
            elif event == "-COUNTERFACTUAL-":
                _counterfactual(window, state, values)
            elif event == "-EXPORT_BATCH_PREDICTIONS-":
                _start_batch_predictions(window, state, values)
            elif event == "-IMPORT_REVIEWED_LABELS-":
                _import_reviewed_labels(window, state, values)
            elif event == "-STORE_MODEL_SLOT-":
                _store_model_slot(window, state, values)
            elif event == "-ACTIVATE_MODEL_SLOT-":
                _activate_model_slot(window, state, values)
            elif event == "-BUILD_ENSEMBLE-":
                _build_ensemble(window, state, values)
            elif event == "-COMPARE_MODELS-":
                _compare_models(window, state, values)
            elif event == "-WEIGHT_ANALYSIS-":
                _run_weight_analysis(window, state, values)
            elif event == "-DISTILL_MODEL-":
                _distill_model(window, state, values)
            elif event == "-MERGE_SLOTS-":
                _merge_slots(window, state, values)
            elif event == "-RUN_MULTI_BACKEND-":
                _start_multi_backend_run(window, state, values)
            elif event == "-PANEL_PREDICT-":
                _panel_predict(window, state, values)
            elif event == "-SAVE_REGISTRY-":
                _save_registry(window, state, values)
            elif event == "-LOAD_REGISTRY-":
                _load_registry(window, state, values)
            elif event == "-BUILD_STACKED_ENSEMBLE-":
                _build_stacked_ensemble(window, state, values)
            elif event == "-TRIAL_DONE-":
                index, total, result = values[event]
                _log(window, f"Trial {index}/{total}: {_format_config(result.config)} | {_format_metrics(result.metrics)}")
            elif event == "-WORKER_DONE-":
                _handle_worker_done(window, state, values[event])
            elif event == "-WORKER_ERROR-":
                state.busy = False
                state.status_message = "Ready"
                _set_busy(window, False)
                _log(window, f"Error: {values[event]}")
        except (DataValidationError, ValueError, RuntimeError, OSError) as exc:
            _log(window, f"Error: {exc}")
        finally:
            _refresh_state(window, state)

    window.close()


def _layout(sg):
    data_column = [
        [sg.Text("Data", font=("Segoe UI", 11, "bold"))],
        [sg.Text("Training sample JSON")],
        [sg.Input(key="-TRAINING_SAMPLE-", expand_x=True)],
        [sg.Button("Add sample", key="-ADD_SAMPLE-"), sg.Button("Clear data", key="-CLEAR_DATA-")],
        [sg.Text("Dataset presets")],
        [
            sg.Combo(preset_labels(), default_value=preset_labels()[0], readonly=True, key="-PRESET_NAME-", expand_x=True),
            sg.Text("Samples"),
            sg.Input("80", key="-PRESET_SAMPLES-", size=(6, 1)),
            sg.Text("Seed"),
            sg.Input("42", key="-PRESET_SEED-", size=(6, 1)),
            sg.Button("Load preset", key="-LOAD_BUILTIN_PRESET-"),
        ],
        [sg.Text("Preset file")],
        [
            sg.Input(key="-PRESET_PATH-", expand_x=True),
            sg.FileBrowse(file_types=(("Preset JSON", "*.json"), ("All files", "*.*"))),
            sg.FileSaveAs(button_text="Choose save", file_types=(("Preset JSON", "*.json"),)),
        ],
        [sg.Input("My preset", key="-PRESET_SAVE_NAME-", size=(22, 1)), sg.Button("Import preset", key="-IMPORT_PRESET-"), sg.Button("Save as preset", key="-SAVE_PRESET-")],
        [sg.Input("", key="-PRESET_DESCRIPTION-", expand_x=True)],
        [sg.Text("CSV dataset")],
        [
            sg.Input(key="-CSV_PATH-", expand_x=True),
            sg.FileBrowse(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
            sg.Button("Load CSV", key="-LOAD_CSV-"),
        ],
        [sg.Text("Dataset JSON")],
        [
            sg.Input(key="-DATASET_PATH-", expand_x=True),
            sg.FileSaveAs(button_text="Choose save", file_types=(("JSON files", "*.json"),)),
            sg.FileBrowse(button_text="Choose load", file_types=(("JSON files", "*.json"), ("All files", "*.*"))),
        ],
        [sg.Button("Save dataset", key="-SAVE_DATASET-"), sg.Button("Load dataset", key="-LOAD_DATASET-")],
    ]

    training_column = [
        [sg.Text("Train & Tune", font=("Segoe UI", 11, "bold"))],
        [
            sg.Text("Samples:"),
            sg.Text("0", key="-SAMPLE_COUNT-", size=(6, 1)),
            sg.Text("Input dim:"),
            sg.Text("-", key="-INPUT_DIM-", size=(6, 1)),
        ],
        [sg.Text("Dataset:"), sg.Text("No data", key="-DATASET_SUMMARY-", expand_x=True)],
        [sg.Text("Model:"), sg.Text("not trained", key="-METRICS_SUMMARY-", expand_x=True)],
        [sg.Text("Audit:"), sg.Text("-", key="-AUDIT_SUMMARY-", expand_x=True)],
        [
            sg.Text("Epochs"),
            sg.Input("50", key="-EPOCHS-", size=(6, 1)),
            sg.Text("Batch"),
            sg.Input("16", key="-BATCH_SIZE-", size=(6, 1)),
            sg.Text("Trials"),
            sg.Input("8", key="-TRIALS-", size=(6, 1)),
            sg.Text("Map"),
            sg.Combo(["linear", "quadratic", "rff"], default_value="rff", readonly=True, key="-FEATURE_MAP-", size=(10, 1)),
            sg.Text("Backend"),
            sg.Combo(
                ["auto"] + available_backends(),
                default_value="auto",
                readonly=True,
                key="-BACKEND-",
                size=(8, 1),
                tooltip="auto: Keras if TF installed else NumPy; mps: tensor-chain classifier",
            ),
        ],
        [
            sg.Text("LR Sched"),
            sg.Combo(["constant", "cosine", "step_decay"], default_value="constant", readonly=True, key="-LR_SCHEDULE-", size=(10, 1)),
            sg.Text("Grad Clip"),
            sg.Input("0.0", key="-GRADIENT_CLIP-", size=(6, 1)),
        ],
        [
            sg.Text("L1 Lasso"),
            sg.Input("0.0", key="-L1_PENALTY-", size=(6, 1)),
            sg.Text("Feat Sel K"),
            sg.Input("", key="-FEATURE_K-", size=(6, 1), tooltip="Leave empty for all features"),
            sg.Checkbox("Enable CV", default=False, key="-USE_CV-"),
            sg.Text("Folds"),
            sg.Input("5", key="-KFOLD_SPLITS-", size=(4, 1)),
        ],
        [
            sg.Checkbox("Enable SMOTE", default=False, key="-USE_SMOTE-", tooltip="Oversample minority class on training split to handle imbalance"),
            sg.Text("SMOTE Neighbors (k)"),
            sg.Input("3", key="-SMOTE_K-", size=(4, 1)),
        ],
        [
            sg.Button("Train once", key="-TRAIN_ONCE-"),
            sg.Button("Run auto experiments", key="-AUTO_EXPERIMENTS-"),
            sg.Button("Weight Analysis", key="-WEIGHT_ANALYSIS-"),
            sg.Text("MPS chi"),
            sg.Input("8", key="-MPS_BOND-", size=(4, 1)),
            sg.Text("phys"),
            sg.Input("4", key="-MPS_PHYS-", size=(4, 1)),
        ],
        [sg.Text("Trial CSV")],
        [
            sg.Input(key="-TRIAL_CSV_PATH-", expand_x=True),
            sg.FileSaveAs(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
        ],
        [sg.Text("Model path")],
        [
            sg.Input(key="-MODEL_PATH-", expand_x=True),
            sg.FileSaveAs(
                button_text="Choose save",
                file_types=(("Italtensor models", "*.italtensor-model.json;*.keras"), ("All files", "*.*")),
            ),
            sg.FileBrowse(
                button_text="Choose load",
                file_types=(("Italtensor models", "*.italtensor-model.json;*.keras"), ("All files", "*.*")),
            ),
        ],
        [sg.Button("Save model", key="-SAVE_MODEL-"), sg.Button("Load model", key="-LOAD_MODEL-")],
        [sg.Text("Report path")],
        [
            sg.Input(key="-REPORT_PATH-", expand_x=True),
            sg.FileSaveAs(file_types=(("Reports", "*.json;*.md"), ("All files", "*.*"))),
            sg.Button("Export report", key="-EXPORT_REPORT-"),
        ],
        [sg.Text("Prediction vector JSON")],
        [
            sg.Input(key="-PREDICTION_VECTOR-", expand_x=True),
            sg.Button("Predict", key="-PREDICT-"),
            sg.Button("Counterfactual", key="-COUNTERFACTUAL-"),
        ],
        [sg.Text("Batch prediction CSV")],
        [
            sg.Input(key="-BATCH_INPUT_PATH-", expand_x=True),
            sg.FileBrowse(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
        ],
        [
            sg.Input(key="-BATCH_OUTPUT_PATH-", expand_x=True),
            sg.FileSaveAs(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
            sg.Button("Export batch predictions", key="-EXPORT_BATCH_PREDICTIONS-"),
            sg.Button("Import reviewed labels", key="-IMPORT_REVIEWED_LABELS-"),
        ],
        [sg.Text("Status:"), sg.Text("Ready", key="-STATUS-", expand_x=True)],
    ]

    slots_column = [
        [sg.Text("Registry & Panel", font=("Segoe UI", 11, "bold"))],
        [
            sg.Text("Fusion"),
            sg.Combo(
                list(FUSION_CHOICES),
                default_value="mean",
                readonly=True,
                key="-PANEL_FUSION-",
                size=(12, 1),
                tooltip="How panel members combine predictions",
            ),
        ],
        [sg.Listbox(values=[], size=(40, 9), key="-MODEL_SLOTS-", enable_events=True, expand_y=True)],
        [
            sg.Button("Store Model", key="-STORE_MODEL_SLOT-", expand_x=True),
            sg.Button("Activate Slot", key="-ACTIVATE_MODEL_SLOT-", expand_x=True),
        ],
        [
            sg.Button("Run Multi-Backend", key="-RUN_MULTI_BACKEND-", expand_x=True),
            sg.Button("Panel Predict", key="-PANEL_PREDICT-", expand_x=True),
        ],
        [
            sg.Button("Build Ensemble", key="-BUILD_ENSEMBLE-", expand_x=True),
            sg.Button("Stacked Ensemble", key="-BUILD_STACKED_ENSEMBLE-", expand_x=True),
        ],
        [
            sg.Button("Compare Models", key="-COMPARE_MODELS-", expand_x=True),
            sg.Button("Slot similarity", key="-SLOT_SIMILARITY-", expand_x=True),
        ],
        [
            sg.Button("Distill Model", key="-DISTILL_MODEL-", expand_x=True),
        ],
        [
            sg.Button("Merge Slots", key="-MERGE_SLOTS-", expand_x=True),
            sg.Button("Save Registry", key="-SAVE_REGISTRY-", expand_x=True),
        ],
        [sg.Button("Load Registry", key="-LOAD_REGISTRY-", expand_x=True)],
        [sg.Text("Registry JSON")],
        [
            sg.Input(key="-REGISTRY_PATH-", expand_x=True),
            sg.FileSaveAs(file_types=(("Registry JSON", "*.json"),)),
            sg.FileBrowse(file_types=(("Registry JSON", "*.json"), ("All files", "*.*"))),
        ],
    ]

    explainability_column = [
        [sg.Text("Explainability & Diagnostics", font=("Segoe UI", 11, "bold"))],
        [sg.Text("Interactive Local Interpretability & Visualizations")],
        [
            sg.Button("SHAP Local Analysis", key="-SHAP_ANALYSIS-", tooltip="Analyze local feature contributions of prediction vector"),
            sg.Button("Visualize Decision Boundary", key="-DECISION_BOUNDARY-", tooltip="Draw PCA-projected 2D decision boundary map of active model"),
        ],
        [sg.HorizontalSeparator()],
        [sg.Text("Automated Model & Dataset Diagnostics")],
        [
            sg.Button("Dataset triage", key="-DATASET_TRIAGE-", expand_x=True),
            sg.Button("Experiment advisor", key="-EXPERIMENT_ADVISOR-", expand_x=True),
            sg.Button("Trial inspector", key="-TRIAL_INSPECTOR-", expand_x=True),
        ],
        [
            sg.Button("Audit dataset", key="-AUDIT_DATASET-", expand_x=True),
            sg.Button("Population drift", key="-POPULATION_DRIFT-", expand_x=True),
            sg.Button("Adversarial validation", key="-ADVERSARIAL_VALIDATION-", expand_x=True),
            sg.Button("Chronological holdout", key="-CHRONOLOGICAL_HOLDOUT-", expand_x=True),
            sg.Button("Learning curve", key="-LEARNING_CURVE-", expand_x=True),
        ],
        [
            sg.Button("Ablation diagnostics", key="-ABLATION_DIAGNOSTICS-", expand_x=True),
            sg.Button("Stress test", key="-STRESS_TEST-", expand_x=True),
        ],
        [
            sg.Button("Slice diagnostics", key="-SLICE_DIAGNOSTICS-", expand_x=True),
            sg.Button("Model response", key="-MODEL_RESPONSE-", expand_x=True),
            sg.Button("Pairwise interactions", key="-PAIRWISE_INTERACTIONS-", expand_x=True),
            sg.Button("Subgroup disparity", key="-SUBGROUP_DISPARITY-", expand_x=True),
            sg.Button("Threshold tradeoff", key="-THRESHOLD_DIAGNOSTICS-", expand_x=True),
        ],
        [
            sg.Button("Decision curve", key="-DECISION_CURVE-", expand_x=True),
            sg.Button("Conformal sets", key="-CONFORMAL_SETS-", expand_x=True),
            sg.Button("Calibration repair", key="-CALIBRATION_REPAIR-", expand_x=True),
            sg.Button("Selective risk", key="-SELECTIVE_RISK-", expand_x=True),
        ],
        [
            sg.Button("Sample review", key="-SAMPLE_REVIEW-", expand_x=True),
            sg.Button("Permutation null", key="-PERMUTATION_NULL-", expand_x=True),
            sg.Button("Dataset cartography", key="-CARTOGRAPHY-", expand_x=True),
        ],
        [
            sg.Button("Reliability diagram", key="-RELIABILITY-", expand_x=True),
            sg.Button("OOD sentinel", key="-OOD_SENTINEL-", expand_x=True),
            sg.Button("Bootstrap stability", key="-BOOTSTRAP_STABILITY-", expand_x=True),
            sg.Button("Prototype audit", key="-PROTOTYPE_AUDIT-", expand_x=True),
            sg.Button("Separability lens", key="-FEATURE_SEPARABILITY-", expand_x=True),
            sg.Button("Neighborhood hardness", key="-NEIGHBORHOOD_HARDNESS-", expand_x=True),
            sg.Button("MPS bond sweep", key="-MPS_BOND_SWEEP-", expand_x=True),
        ],
        [
            sg.Button("Export trials CSV", key="-EXPORT_TRIALS-", expand_x=True),
        ],
    ]

    tab_data = sg.Tab("Workspace & Data", data_column, key="-TAB_DATA-")
    tab_training = sg.Tab("Training & Tuning", training_column, key="-TAB_TRAINING-")
    tab_slots = sg.Tab("Model Slots Registry", slots_column, key="-TAB_SLOTS-")
    tab_explainability = sg.Tab("Explainability & Diagnostics", explainability_column, key="-TAB_EXPLAINABILITY-")

    tab_group = sg.TabGroup([[tab_data, tab_training, tab_slots, tab_explainability]], expand_x=True, expand_y=True)

    return [
        [tab_group],
        [
            sg.Multiline(
                size=(110, 18),
                key="-LOG-",
                autoscroll=True,
                disabled=True,
                expand_x=True,
                expand_y=True,
                background_color="#14151A",
                text_color="#E4E6EB",
            )
        ],
        [sg.Button("Exit")],
    ]


def _add_sample(window, state: AppState, values: dict[str, Any]) -> None:
    features, label = parse_training_example(values["-TRAINING_SAMPLE-"], state.input_dim)
    _invalidate_model_artifacts(state)
    if state.input_dim is None:
        state.input_dim = len(features)
    state.features.append(features)
    state.labels.append(label)
    _log(window, f"Added sample {len(state.labels)} with label {label}.")


def _load_builtin_preset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = generate_builtin_preset(
        values["-PRESET_NAME-"],
        sample_count=_positive_int(values["-PRESET_SAMPLES-"], "preset samples"),
        seed=_int_value(values["-PRESET_SEED-"], "preset seed"),
    )
    _replace_dataset(state, dataset, window=window)
    metadata = preset_metadata(values["-PRESET_NAME-"])
    _apply_preset_metadata(window, metadata)
    _log(window, f"Loaded preset '{values['-PRESET_NAME-']}' with {dataset.sample_count} samples.")
    _log(window, _format_preset_metadata(metadata))


def _import_preset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset, metadata = load_preset_file(_required_path(values["-PRESET_PATH-"], "preset path"))
    _replace_dataset(state, dataset, window=window)
    _apply_preset_metadata(window, metadata)
    _log(window, f"Imported preset '{metadata['name']}' with {dataset.sample_count} samples.")
    _log(window, _format_preset_metadata(metadata))


def _save_preset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = validate_dataset(state.features, state.labels, min_samples=1)
    path = save_preset_file(
        _required_path(values["-PRESET_PATH-"], "preset path"),
        dataset,
        name=values["-PRESET_SAVE_NAME-"],
        description=values["-PRESET_DESCRIPTION-"],
        training_defaults=_preset_training_defaults_from_values(values),
        recommended_feature_map=values.get("-FEATURE_MAP-", "linear"),
        prediction_examples=_preset_prediction_examples_from_values(values, dataset.input_dim),
    )
    _log(window, f"Saved preset '{values['-PRESET_SAVE_NAME-']}' to {path}.")


def _preset_training_defaults_from_values(values: dict[str, Any]) -> dict[str, object]:
    defaults: dict[str, object] = {
        "epochs": _positive_int(values.get("-EPOCHS-", "50"), "epochs"),
        "batch_size": _positive_int(values.get("-BATCH_SIZE-", "16"), "batch size"),
        "trials": _positive_int(values.get("-TRIALS-", "8"), "trials"),
        "feature_map": values.get("-FEATURE_MAP-", "linear"),
        "backend": values.get("-BACKEND-", "auto"),
        "lr_schedule": values.get("-LR_SCHEDULE-", "constant"),
        "gradient_clip": _nonnegative_float(values.get("-GRADIENT_CLIP-", "0.0"), "gradient clip"),
        "l1_penalty": _nonnegative_float(values.get("-L1_PENALTY-", "0.0"), "L1 penalty"),
        "mps_bond_dim": _positive_int(values.get("-MPS_BOND-", "8"), "MPS bond dimension"),
        "mps_physical_dim": _positive_int(values.get("-MPS_PHYS-", "4"), "MPS physical dimension"),
    }
    feature_k = str(values.get("-FEATURE_K-", "")).strip()
    if feature_k:
        defaults["feature_selection_k"] = _positive_int(feature_k, "feature selection k")
    return defaults


def _preset_prediction_examples_from_values(values: dict[str, Any], input_dim: int) -> list[dict[str, object]]:
    raw = str(values.get("-PREDICTION_VECTOR-", "")).strip()
    if not raw:
        return []
    try:
        vector = parse_prediction_vector(raw, input_dim)
    except DataValidationError:
        return []
    return [{"name": "Saved prediction vector", "features": vector, "expected_label": None}]


def _load_csv(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = load_csv_dataset(_required_path(values["-CSV_PATH-"], "CSV path"))
    _replace_dataset(state, dataset, window=window)
    _log(window, f"Loaded {dataset.sample_count} samples from CSV.")


def _save_dataset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = validate_dataset(state.features, state.labels, min_samples=1)
    path = save_dataset(_required_path(values["-DATASET_PATH-"], "dataset path"), dataset)
    _log(window, f"Saved dataset to {path}.")


def _load_dataset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = load_dataset(_required_path(values["-DATASET_PATH-"], "dataset path"))
    _replace_dataset(state, dataset, window=window)
    _log(window, f"Loaded {dataset.sample_count} samples from dataset JSON.")


def _clear_data(window, state: AppState) -> None:
    state.features.clear()
    state.labels.clear()
    state.input_dim = None
    _invalidate_model_artifacts(state)
    window["-AUDIT_SUMMARY-"].update("-")
    _log(window, "Cleared dataset.")


def _audit_dataset(window, state: AppState) -> None:
    if len(state.labels) < 1:
        raise ValueError("Add data before running an audit.")
    summary = format_audit_summary(audit_dataset(state.features, state.labels))
    window["-AUDIT_SUMMARY-"].update(summary[:120] + ("..." if len(summary) > 120 else ""))
    _log(window, summary)


def _start_learning_curve(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=8, require_two_classes=True)
    config = _config_from_values(values)

    def task() -> tuple[str, list[dict[str, Any]]]:
        return "learning_curve", learning_curve_points(dataset.features, dataset.labels, config)

    _start_worker(window, state, "Computing learning curve...", task)


def _start_ablation_diagnostics(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running ablation diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_ablation_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "ablation_diagnostics", report

    _start_worker(window, state, "Running feature ablation diagnostics...", task)


def _start_decision_curve(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running decision curve diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_decision_curve_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            current_threshold=state.latest_threshold,
        )
        return "decision_curve", report

    _start_worker(window, state, "Running decision curve diagnostics...", task)


def _start_conformal_sets(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running conformal set diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=2, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_conformal_set_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
        )
        return "conformal_sets", report

    _start_worker(window, state, "Running conformal set diagnostics...", task)


def _start_calibration_repair(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running calibration repair diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_calibration_repair_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
        )
        return "calibration_repair", report

    _start_worker(window, state, "Running calibration repair diagnostics...", task)


def _start_selective_risk(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running selective risk diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_selective_risk_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "selective_risk", report

    _start_worker(window, state, "Running selective risk diagnostics...", task)


def _start_stress_test(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running a stress test.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_stress_suite(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "stress_test", report

    _start_worker(window, state, "Running robustness stress suite...", task)


def _start_slice_diagnostics(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running slice diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_slice_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "slice_diagnostics", report

    _start_worker(window, state, "Running slice diagnostics...", task)


def _start_model_response(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running model response diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_model_response_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
        )
        return "model_response", report

    _start_worker(window, state, "Running model response diagnostics...", task)


def _start_pairwise_interactions(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running pairwise interaction diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_pairwise_interaction_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
        )
        return "pairwise_interactions", report

    _start_worker(window, state, "Running pairwise interaction diagnostics...", task)


def _start_subgroup_disparity(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running subgroup disparity diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=2, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_subgroup_disparity_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "subgroup_disparity", report

    _start_worker(window, state, "Running subgroup disparity diagnostics...", task)


def _start_threshold_diagnostics(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running threshold diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_threshold_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            current_threshold=state.latest_threshold,
        )
        return "threshold_diagnostics", report

    _start_worker(window, state, "Running threshold tradeoff sweep...", task)


def _start_sample_review(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running sample review.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_sample_review(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "sample_review", report

    _start_worker(window, state, "Running sample review...", task)


def _start_population_drift(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=6, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_population_drift_diagnostics(dataset.features, dataset.labels)
        return "population_drift", report

    _start_worker(window, state, "Running population drift diagnostics...", task)


def _start_adversarial_validation(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=12, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_adversarial_validation_diagnostics(dataset.features, dataset.labels)
        return "adversarial_validation", report

    _start_worker(window, state, "Running adversarial validation...", task)


def _start_chronological_holdout(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=16, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_chronological_holdout_diagnostics(
            dataset.features,
            dataset.labels,
            threshold=state.latest_threshold,
        )
        return "chronological_holdout", report

    _start_worker(window, state, "Running chronological holdout replay...", task)


def _start_permutation_null(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running permutation-null diagnostics.")
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_permutation_null_diagnostics(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "permutation_null", report

    _start_worker(window, state, "Running permutation-null diagnostics...", task)


def _config_from_values(values: dict[str, Any]) -> ModelConfig:
    l1_raw = values.get("-L1_PENALTY-", "0.0").strip()
    l1_penalty = float(l1_raw) if l1_raw else 0.0
    feat_k_raw = values.get("-FEATURE_K-", "").strip()
    feature_selection_k = int(feat_k_raw) if feat_k_raw else None
    grad_clip_raw = values.get("-GRADIENT_CLIP-", "0.0").strip()
    try:
        gradient_clip = float(grad_clip_raw) if grad_clip_raw else 0.0
    except ValueError:
        raise ValueError("Gradient clip must be a float.")
    mps_bond_raw = values.get("-MPS_BOND-", "8").strip()
    try:
        mps_bond_dim = int(mps_bond_raw) if mps_bond_raw else 8
    except ValueError as exc:
        raise ValueError("MPS bond dimension must be an integer.") from exc
    if mps_bond_dim < 2:
        raise ValueError("MPS bond dimension must be at least 2.")
    mps_phys_raw = values.get("-MPS_PHYS-", "4").strip()
    try:
        mps_physical_dim = int(mps_phys_raw) if mps_phys_raw else 4
    except ValueError as exc:
        raise ValueError("MPS physical dimension must be an integer.") from exc
    if mps_physical_dim < 2:
        raise ValueError("MPS physical dimension must be at least 2.")
    return ModelConfig(
        hidden_layers=(32,),
        learning_rate=0.001,
        batch_size=_positive_int(values["-BATCH_SIZE-"], "batch size"),
        max_epochs=_positive_int(values["-EPOCHS-"], "epochs"),
        feature_map=values["-FEATURE_MAP-"],
        l1_penalty=l1_penalty,
        feature_selection_k=feature_selection_k,
        lr_schedule=values.get("-LR_SCHEDULE-", "constant"),
        gradient_clip=gradient_clip,
        backend=values.get("-BACKEND-", "auto"),
        mps_bond_dim=mps_bond_dim,
        mps_physical_dim=mps_physical_dim,
    )


def _start_train_once(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)
    config = _config_from_values(values)

    use_smote = values.get("-USE_SMOTE-", False)
    smote_k = _positive_int(values.get("-SMOTE_K-", "3"), "SMOTE k")

    use_cv = values.get("-USE_CV-", False)
    if use_cv:
        n_splits = _positive_int(values.get("-KFOLD_SPLITS-", "5"), "CV folds")
        def task() -> tuple[str, ExperimentResult]:
            return "single", train_single_model_cv(
                dataset.features, dataset.labels, config, n_splits=n_splits, use_smote=use_smote, smote_k=smote_k
            )
        _start_worker(window, state, f"Training with {n_splits}-Fold CV...", task)
    else:
        def task() -> tuple[str, ExperimentResult]:
            return "single", train_single_model(
                dataset.features, dataset.labels, config, use_smote=use_smote, smote_k=smote_k
            )
        _start_worker(window, state, "Training one model...", task)


def _start_auto_experiments(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)
    trials = _positive_int(values["-TRIALS-"], "trials")

    def task() -> tuple[str, list[ExperimentResult]]:
        results = run_experiments(
            dataset.features,
            dataset.labels,
            trials=trials,
            progress_callback=lambda index, total, result: window.write_event_value(
                "-TRIAL_DONE-", (index, total, result)
            ),
        )
        return "experiments", results

    _start_worker(window, state, f"Running {trials} auto experiment(s)...", task)


def _save_model(window, state: AppState, values: dict[str, Any]) -> None:
    if state.model is None or state.input_dim is None:
        raise ValueError("Train or load a model before saving.")
    model_path, metadata_path = save_model_bundle(
        state.model,
        _required_path(values["-MODEL_PATH-"], "model path"),
        input_dim=state.input_dim,
        config=state.latest_config or ModelConfig(),
        metrics=state.latest_metrics,
        threshold=state.latest_threshold,
        preprocessor=state.preprocessor,
        feature_importances=state.feature_importances,
        trial_history=state.trial_history,
        uncertainty_metadata=state.uncertainty_metadata,
        ablation_report=state.latest_ablation_report,
        decision_curve_report=state.latest_decision_curve_report,
        conformal_set_report=state.latest_conformal_set_report,
        calibration_repair_report=state.latest_calibration_repair_report,
        selective_risk_report=state.latest_selective_risk_report,
        sample_review_report=state.latest_sample_review_report,
        threshold_report=state.latest_threshold_report,
        model_response_report=state.latest_model_response_report,
        pairwise_interaction_report=state.latest_pairwise_interaction_report,
        slice_report=state.latest_slice_report,
        subgroup_disparity_report=state.latest_subgroup_disparity_report,
        stress_report=state.latest_stress_report,
        permutation_null_report=state.latest_permutation_null_report,
        population_drift_report=state.latest_population_drift_report,
        adversarial_validation_report=state.latest_adversarial_validation_report,
        chronological_holdout_report=state.latest_chronological_holdout_report,
        cartography_report=state.latest_cartography_report,
        ood_sentinel_report=state.latest_ood_sentinel_report,
        bootstrap_stability_report=state.latest_bootstrap_stability_report,
        prototype_audit_report=state.latest_prototype_audit_report,
        feature_separability_report=state.latest_feature_separability_report,
        neighborhood_hardness_report=state.latest_neighborhood_hardness_report,
        dataset_triage_report=state.latest_dataset_triage_report,
        experiment_advisor_report=state.latest_experiment_advisor_report,
        trial_inspector_report=state.latest_trial_inspector_report,
        mps_sweep_report=state.latest_mps_sweep_report,
    )
    window["-MODEL_PATH-"].update(str(model_path))
    _log(window, f"Saved model to {model_path} and metadata to {metadata_path}.")


def _load_model(window, state: AppState, values: dict[str, Any]) -> None:
    model, metadata = load_model_bundle(_required_path(values["-MODEL_PATH-"], "model path"))
    input_dim = metadata.get("input_dim")
    if input_dim is None:
        input_dim = getattr(model, "input_dim", None)
    if input_dim is None:
        model_input = getattr(model, "input_shape", None)
        input_dim = int(model_input[-1]) if model_input else None
    if input_dim is None:
        raise ValueError("Loaded model metadata does not include input_dim.")
    if state.input_dim is not None and int(input_dim) != state.input_dim:
        raise ValueError(f"Loaded model expects {input_dim} features, current dataset uses {state.input_dim}.")

    state.model = model
    state.input_dim = int(input_dim)
    best_config = metadata.get("best_config")
    state.latest_config = ModelConfig.from_dict(best_config) if isinstance(best_config, dict) else None
    metrics = metadata.get("validation_metrics")
    state.latest_metrics = metrics if isinstance(metrics, dict) else {}
    state.latest_threshold = float(metadata.get("threshold", 0.5))
    preprocessing = metadata.get("preprocessing")
    state.preprocessor = FeatureStandardizer.from_dict(
        preprocessing if isinstance(preprocessing, dict) else None,
        input_dim=state.input_dim,
    )
    importances = metadata.get("feature_importances")
    state.feature_importances = importances if isinstance(importances, list) else []
    trial_history = metadata.get("trial_history")
    state.trial_history = trial_history if isinstance(trial_history, list) else []
    uncertainty = metadata.get("uncertainty")
    state.uncertainty_metadata = uncertainty if isinstance(uncertainty, dict) else {}
    ablation_report = metadata.get("feature_ablation_diagnostics")
    decision_curve_report = metadata.get("decision_curve_diagnostics")
    conformal_set_report = metadata.get("posthoc_conformal_diagnostics") or metadata.get("conformal_set_diagnostics")
    calibration_repair_report = metadata.get("posthoc_calibration_repair_diagnostics")
    selective_risk_report = metadata.get("selective_prediction_diagnostics")
    sample_review_report = metadata.get("sample_review")
    threshold_report = metadata.get("threshold_diagnostics")
    model_response_report = metadata.get("model_response_diagnostics")
    pairwise_interaction_report = metadata.get("pairwise_interaction_diagnostics")
    slice_report = metadata.get("slice_diagnostics")
    subgroup_disparity_report = metadata.get("subgroup_disparity_diagnostics")
    stress_report = metadata.get("stress_lab")
    permutation_null_report = metadata.get("posthoc_permutation_null_diagnostics")
    population_drift_report = metadata.get("population_drift_diagnostics")
    adversarial_validation_report = metadata.get("adversarial_validation_diagnostics")
    chronological_holdout_report = metadata.get("chronological_holdout_diagnostics")
    cartography_report = metadata.get("dataset_cartography")
    ood_sentinel_report = metadata.get("ood_sentinel")
    bootstrap_stability_report = metadata.get("bootstrap_stability_diagnostics")
    prototype_audit_report = metadata.get("prototype_audit")
    feature_separability_report = metadata.get("feature_separability")
    neighborhood_hardness_report = metadata.get("neighborhood_hardness")
    dataset_triage_report = metadata.get("dataset_triage")
    experiment_advisor_report = metadata.get("experiment_advisor")
    trial_inspector_report = metadata.get("trial_inspector")
    mps_sweep_report = metadata.get("mps_bond_sweep")
    state.latest_ablation_report = ablation_report if isinstance(ablation_report, dict) else None
    state.latest_decision_curve_report = decision_curve_report if isinstance(decision_curve_report, dict) else None
    state.latest_conformal_set_report = conformal_set_report if isinstance(conformal_set_report, dict) else None
    state.latest_calibration_repair_report = (
        calibration_repair_report if isinstance(calibration_repair_report, dict) else None
    )
    state.latest_selective_risk_report = selective_risk_report if isinstance(selective_risk_report, dict) else None
    state.latest_sample_review_report = sample_review_report if isinstance(sample_review_report, dict) else None
    state.latest_threshold_report = threshold_report if isinstance(threshold_report, dict) else None
    state.latest_model_response_report = model_response_report if isinstance(model_response_report, dict) else None
    state.latest_pairwise_interaction_report = (
        pairwise_interaction_report if isinstance(pairwise_interaction_report, dict) else None
    )
    state.latest_slice_report = slice_report if isinstance(slice_report, dict) else None
    state.latest_subgroup_disparity_report = (
        subgroup_disparity_report if isinstance(subgroup_disparity_report, dict) else None
    )
    state.latest_stress_report = stress_report if isinstance(stress_report, dict) else None
    state.latest_permutation_null_report = (
        permutation_null_report if isinstance(permutation_null_report, dict) else None
    )
    state.latest_population_drift_report = (
        population_drift_report if isinstance(population_drift_report, dict) else None
    )
    state.latest_adversarial_validation_report = (
        adversarial_validation_report if isinstance(adversarial_validation_report, dict) else None
    )
    state.latest_chronological_holdout_report = (
        chronological_holdout_report if isinstance(chronological_holdout_report, dict) else None
    )
    state.latest_cartography_report = cartography_report if isinstance(cartography_report, dict) else None
    state.latest_ood_sentinel_report = ood_sentinel_report if isinstance(ood_sentinel_report, dict) else None
    state.latest_bootstrap_stability_report = (
        bootstrap_stability_report if isinstance(bootstrap_stability_report, dict) else None
    )
    state.latest_prototype_audit_report = prototype_audit_report if isinstance(prototype_audit_report, dict) else None
    state.latest_feature_separability_report = (
        feature_separability_report if isinstance(feature_separability_report, dict) else None
    )
    state.latest_neighborhood_hardness_report = (
        neighborhood_hardness_report if isinstance(neighborhood_hardness_report, dict) else None
    )
    state.latest_dataset_triage_report = dataset_triage_report if isinstance(dataset_triage_report, dict) else None
    if state.latest_dataset_triage_report is not None:
        state.latest_feature_separability_report = state.latest_feature_separability_report or _dict_or_none(
            state.latest_dataset_triage_report.get("feature_separability")
        )
        state.latest_prototype_audit_report = state.latest_prototype_audit_report or _dict_or_none(
            state.latest_dataset_triage_report.get("prototype_audit")
        )
        state.latest_neighborhood_hardness_report = state.latest_neighborhood_hardness_report or _dict_or_none(
            state.latest_dataset_triage_report.get("neighborhood_hardness")
        )
        state.latest_ood_sentinel_report = state.latest_ood_sentinel_report or _dict_or_none(
            state.latest_dataset_triage_report.get("ood_sentinel")
        )
    state.latest_experiment_advisor_report = (
        experiment_advisor_report if isinstance(experiment_advisor_report, dict) else None
    )
    state.latest_trial_inspector_report = trial_inspector_report if isinstance(trial_inspector_report, dict) else None
    state.latest_mps_sweep_report = mps_sweep_report if isinstance(mps_sweep_report, dict) else None
    _log(window, f"Loaded model expecting {state.input_dim} features.")


def _export_report(window, state: AppState, values: dict[str, Any]) -> None:
    if state.latest_config is None and not state.latest_metrics and not state.labels:
        raise ValueError("Load a dataset or train/load a model before exporting a report.")
    report = build_experiment_report(
        sample_count=len(state.labels),
        input_dim=state.input_dim,
        labels=state.labels,
        features=state.features,
        config=state.latest_config,
        metrics=state.latest_metrics,
        threshold=state.latest_threshold,
        preprocessor=state.preprocessor,
        feature_importances=state.feature_importances,
        trial_history=state.trial_history,
        uncertainty_metadata=state.uncertainty_metadata,
        ablation_report=state.latest_ablation_report,
        decision_curve_report=state.latest_decision_curve_report,
        conformal_set_report=state.latest_conformal_set_report,
        calibration_repair_report=state.latest_calibration_repair_report,
        selective_risk_report=state.latest_selective_risk_report,
        sample_review_report=state.latest_sample_review_report,
        threshold_report=state.latest_threshold_report,
        model_response_report=state.latest_model_response_report,
        pairwise_interaction_report=state.latest_pairwise_interaction_report,
        slice_report=state.latest_slice_report,
        subgroup_disparity_report=state.latest_subgroup_disparity_report,
        stress_report=state.latest_stress_report,
        permutation_null_report=state.latest_permutation_null_report,
        population_drift_report=state.latest_population_drift_report,
        adversarial_validation_report=state.latest_adversarial_validation_report,
        chronological_holdout_report=state.latest_chronological_holdout_report,
        cartography_report=state.latest_cartography_report,
        ood_sentinel_report=state.latest_ood_sentinel_report,
        bootstrap_stability_report=state.latest_bootstrap_stability_report,
        prototype_audit_report=state.latest_prototype_audit_report,
        feature_separability_report=state.latest_feature_separability_report,
        neighborhood_hardness_report=state.latest_neighborhood_hardness_report,
        dataset_triage_report=state.latest_dataset_triage_report,
        experiment_advisor_report=state.latest_experiment_advisor_report,
        trial_inspector_report=state.latest_trial_inspector_report,
        mps_sweep_report=state.latest_mps_sweep_report,
    )
    path = export_experiment_report(_required_path(values["-REPORT_PATH-"], "report path"), report)
    _log(window, f"Exported report to {path}.")


def _predict(window, state: AppState, values: dict[str, Any]) -> None:
    if state.model is None:
        raise ValueError("Train or load a model before predicting.")
    vector = parse_prediction_vector(values["-PREDICTION_VECTOR-"], state.input_dim)
    if state.preprocessor is not None:
        prepared = state.preprocessor.transform(vector)
    else:
        prepared = vector
    probability = float(predict_probability(state.model, prepared)[0])
    label = 1 if probability >= state.latest_threshold else 0
    uncertainty_note = _format_uncertainty_prediction(probability, state.uncertainty_metadata)
    _log(
        window,
        f"Prediction: label={label}, probability={probability:.4f}, threshold={state.latest_threshold:.4f}{uncertainty_note}",
    )


def _counterfactual(window, state: AppState, values: dict[str, Any]) -> None:
    if state.model is None:
        raise ValueError("Train or load a model before finding a counterfactual.")
    vector = parse_prediction_vector(values["-PREDICTION_VECTOR-"], state.input_dim)
    result = find_counterfactual(
        state.model,
        vector,
        preprocessor=state.preprocessor,
        threshold=state.latest_threshold,
    )
    _log(window, format_counterfactual_result(result))


def _start_batch_predictions(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    if state.model is None or state.input_dim is None:
        raise ValueError("Train or load a model before exporting batch predictions.")
    input_path = _required_path(values["-BATCH_INPUT_PATH-"], "batch prediction input CSV")
    output_path = _required_path(values["-BATCH_OUTPUT_PATH-"], "batch prediction output CSV")

    def task() -> tuple[str, tuple[str, int]]:
        path, count = score_prediction_csv(
            state.model,
            input_path,
            output_path,
            expected_dim=state.input_dim,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
            uncertainty_metadata=state.uncertainty_metadata,
        )
        return "batch_predictions", (str(path), count)

    _start_worker(window, state, "Exporting batch predictions...", task)


def _import_reviewed_labels(window, state: AppState, values: dict[str, Any]) -> None:
    path = _required_path(values["-BATCH_OUTPUT_PATH-"], "reviewed batch prediction CSV")
    table, reviewed_labels = load_reviewed_prediction_csv(path, state.input_dim)
    combined_features = state.features + table.features.astype(float).tolist()
    combined_labels = state.labels + reviewed_labels.astype(int).tolist()
    dataset = validate_dataset(combined_features, combined_labels, min_samples=1, require_two_classes=False)
    _replace_dataset(state, dataset)
    _log(window, f"Imported {reviewed_labels.shape[0]} reviewed label(s) from {path}. Current dataset: {len(state.labels)} samples.")


def _handle_worker_done(window, state: AppState, payload: tuple[str, Any]) -> None:
    kind, result = payload
    state.busy = False
    state.status_message = "Ready"
    _set_busy(window, False)

    if kind == "single":
        training_result = result
        state.model = training_result.model
        state.latest_config = training_result.config
        state.latest_metrics = training_result.metrics
        state.latest_threshold = training_result.threshold
        state.preprocessor = training_result.preprocessor
        state.feature_importances = training_result.feature_importances
        state.trial_history = [_summarize_trial(training_result)]
        state.uncertainty_metadata = training_result.uncertainty
        state.latest_ablation_report = None
        state.latest_decision_curve_report = None
        state.latest_conformal_set_report = None
        state.latest_calibration_repair_report = None
        state.latest_selective_risk_report = None
        state.latest_sample_review_report = None
        state.latest_threshold_report = None
        state.latest_model_response_report = None
        state.latest_pairwise_interaction_report = None
        state.latest_slice_report = None
        state.latest_subgroup_disparity_report = None
        state.latest_stress_report = None
        state.latest_permutation_null_report = None
        state.latest_population_drift_report = None
        state.latest_adversarial_validation_report = None
        state.latest_chronological_holdout_report = None
        state.latest_cartography_report = None
        state.latest_ood_sentinel_report = None
        state.latest_bootstrap_stability_report = None
        state.latest_experiment_advisor_report = None
        state.latest_trial_inspector_report = None
        state.latest_mps_sweep_report = None
        _log(window, f"Training complete: {_format_metrics(training_result.metrics)}")
        _log(window, _format_calibration(training_result.metrics))
        _log(window, _format_uncertainty(training_result.uncertainty))
        _log(window, _format_cv_summary(training_result.metrics))
        _log(window, _format_importances(training_result.feature_importances))
    elif kind == "experiments":
        best = select_best_result(result)
        state.model = best.model
        state.latest_config = best.config
        state.latest_metrics = best.metrics
        state.latest_threshold = best.threshold
        state.preprocessor = best.preprocessor
        state.feature_importances = best.feature_importances
        state.trial_history = [_summarize_trial(item) for item in result]
        state.uncertainty_metadata = best.uncertainty
        state.latest_ablation_report = None
        state.latest_decision_curve_report = None
        state.latest_conformal_set_report = None
        state.latest_calibration_repair_report = None
        state.latest_selective_risk_report = None
        state.latest_sample_review_report = None
        state.latest_threshold_report = None
        state.latest_model_response_report = None
        state.latest_pairwise_interaction_report = None
        state.latest_slice_report = None
        state.latest_subgroup_disparity_report = None
        state.latest_stress_report = None
        state.latest_permutation_null_report = None
        state.latest_population_drift_report = None
        state.latest_adversarial_validation_report = None
        state.latest_chronological_holdout_report = None
        state.latest_cartography_report = None
        state.latest_ood_sentinel_report = None
        state.latest_bootstrap_stability_report = None
        state.latest_experiment_advisor_report = None
        state.latest_trial_inspector_report = None
        state.latest_mps_sweep_report = None
        _log(window, f"Best config: {_format_config(best.config)}")
        _log(window, f"Best metrics: {_format_metrics(best.metrics)}")
        _log(window, _format_calibration(best.metrics))
        _log(window, _format_uncertainty(best.uncertainty))
        _log(window, _format_importances(best.feature_importances))
    elif kind == "batch_predictions":
        path, count = result
        _log(window, f"Exported {count} batch prediction(s) to {path}.")
    elif kind == "learning_curve":
        for point in result:
            _log(
                window,
                f"Curve fraction={point['train_fraction']:.2f} "
                f"samples={point['train_samples']} "
                f"F1={point['f1']:.4f} acc={point['accuracy']:.4f}",
            )
    elif kind == "ablation_diagnostics":
        state.latest_ablation_report = result
        _log(window, format_ablation_summary(result))
        for item in result.get("features", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}: "
                f"drop={float(item['f1_drop']):.4f}, "
                f"perm_drop={float(item['permutation_f1_drop']):.4f}, "
                f"flip={max(float(item['label_flip_rate']), float(item['permutation_label_flip_rate'])):.4f}, "
                f"corr={float(item['label_correlation']):.4f}, "
                f"flags={flags}",
            )
    elif kind == "decision_curve":
        state.latest_decision_curve_report = result
        _log(window, format_decision_curve_summary(result))
        for item in result.get("points", [])[:6]:
            _log(
                window,
                f"  t={float(item['threshold']):.4f}: "
                f"model_nb={float(item['net_benefit_model']):.4f}, "
                f"all_nb={float(item['net_benefit_treat_all']):.4f}, "
                f"gain={float(item['delta_vs_best_default']):.4f}, "
                f"default={item['best_default_strategy']}",
            )
    elif kind == "conformal_sets":
        state.latest_conformal_set_report = result
        _log(window, format_conformal_set_summary(result))
        for item in result.get("points", [])[:6]:
            singleton_accuracy = item.get("singleton_accuracy")
            singleton_text = "-" if singleton_accuracy is None else f"{float(singleton_accuracy):.4f}"
            _log(
                window,
                f"  alpha={float(item['alpha']):.4f}: "
                f"target={float(item['target_coverage']):.4f}, "
                f"coverage={float(item['empirical_coverage']):.4f}, "
                f"mean_size={float(item['mean_set_size']):.4f}, "
                f"singleton_acc={singleton_text}",
            )
    elif kind == "calibration_repair":
        state.latest_calibration_repair_report = result
        _log(window, format_calibration_repair_summary(result))
        for item in result.get("methods", [])[:6]:
            _log(
                window,
                f"  {item['method']}: "
                f"Brier={float(item['brier_score']):.4f}, "
                f"ECE={float(item['ece']):.4f}, "
                f"logloss={float(item['log_loss']):.4f}, "
                f"dBrier={float(item.get('brier_improvement', 0.0)):.4f}",
            )
    elif kind == "selective_risk":
        state.latest_selective_risk_report = result
        _log(window, format_selective_risk_summary(result))
        for item in result.get("ranked_cutoffs", [])[:6]:
            _log(
                window,
                f"  cutoff={float(item['confidence_cutoff']):.4f}: "
                f"coverage={float(item['coverage']):.4f}, "
                f"risk={float(item['error_rate']):.4f}, "
                f"acc={float(item['accuracy']):.4f}, "
                f"F1={float(item['f1']):.4f}",
            )
    elif kind == "stress_test":
        state.latest_stress_report = result
        _log(window, format_stress_summary(result))
        for item in result.get("perturbations", [])[:6]:
            label = item["kind"]
            if "feature_index" in item:
                label = f"{label}[x{int(item['feature_index']) + 1}]"
            _log(
                window,
                f"  {label}@{float(item['level']):.2f}: "
                f"F1={float(item['f1']):.4f}, "
                f"flip={float(item['label_flip_rate']):.4f}, "
                f"prob_shift={float(item['mean_probability_shift']):.4f}",
            )
    elif kind == "slice_diagnostics":
        state.latest_slice_report = result
        _log(window, format_slice_summary(result))
        for item in result.get("slices", [])[:6]:
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}[{float(item['left']):.4g}, {float(item['right']):.4g}]: "
                f"n={int(item['count'])}, "
                f"F1={float(item['f1']):.4f}, "
                f"acc={float(item['accuracy']):.4f}, "
                f"delta={float(item['f1_delta']):.4f}",
            )
    elif kind == "model_response":
        state.latest_model_response_report = result
        _log(window, format_model_response_summary(result))
        for item in result.get("features", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}: "
                f"range={float(item['response_range']):.4f}, "
                f"change={float(item['signed_change']):.4f}, "
                f"direction={item['direction']}, "
                f"flags={flags}",
            )
    elif kind == "pairwise_interactions":
        state.latest_pairwise_interaction_report = result
        _log(window, format_pairwise_interaction_summary(result))
        for item in result.get("pairs", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_i']) + 1}:x{int(item['feature_j']) + 1}: "
                f"H={float(item['interaction_strength']):.4f}, "
                f"max_abs={float(item['max_abs_interaction']):.4f}, "
                f"cross={int(item['threshold_crossings'])}, "
                f"flags={flags}",
            )
    elif kind == "subgroup_disparity":
        state.latest_subgroup_disparity_report = result
        _log(window, format_subgroup_disparity_summary(result))
        for item in result.get("subgroups", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  {item['label']}: "
                f"n={int(item['count'])}, "
                f"gap={float(item['risk_score']):.4f}, "
                f"metric={item['worst_metric']}, "
                f"flags={flags}",
            )
    elif kind == "threshold_diagnostics":
        state.latest_threshold_report = result
        _log(window, format_threshold_summary(result))
        for label, item in (
            ("best_f1", result.get("best_f1")),
            ("best_balanced", result.get("best_balanced_accuracy")),
            ("min_cost", result.get("min_cost")),
        ):
            if isinstance(item, dict):
                _log(
                    window,
                    f"  {label}: t={float(item['threshold']):.4f}, "
                    f"F1={float(item['f1']):.4f}, "
                    f"precision={float(item['precision']):.4f}, "
                    f"recall={float(item['recall']):.4f}, "
                    f"cost={float(item['cost']):.4f}",
                )

    elif kind == "cartography":
        state.latest_cartography_report = result
        _log(window, format_cartography_summary(result))
        for region_name in ("overconfident_wrong", "ambiguous", "hard_to_learn", "easy_to_learn"):
            for item in result.get("regions", {}).get(region_name, [])[:2]:
                _log(
                    window,
                    f"  {region_name} row={int(item['row_index'])}: "
                    f"label={int(item['label'])}, conf={float(item['confidence']):.3f}, "
                    f"var={float(item['variability']):.3f}",
                )
    elif kind == "ood_sentinel":
        state.latest_ood_sentinel_report = result
        _log(window, format_ood_sentinel_summary(result))
        for item in result.get("rows", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            row_text = f"  row={int(item['row_index'])}: score={float(item['ood_score']):.4f}, "
            row_text += (
                f"max_z={float(item['max_abs_robust_z']):.4f}, "
                f"nn={float(item['nearest_neighbor_distance']):.4f}"
            )
            if item.get("loss") is not None:
                row_text += f", loss={float(item['loss']):.4f}, p={float(item['probability']):.4f}"
            _log(window, row_text + f", flags={flags}")
    elif kind == "bootstrap_stability":
        state.latest_bootstrap_stability_report = result
        _log(window, format_bootstrap_stability_summary(result))
        metrics = result.get("ensemble_metrics", {})
        if metrics:
            _log(
                window,
                "  ensemble: "
                f"F1={float(metrics.get('f1', 0.0)):.4f}, "
                f"acc={float(metrics.get('accuracy', 0.0)):.4f}, "
                f"Brier={float(metrics.get('brier_score', 0.0)):.4f}",
            )
        for item in result.get("rows", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  row={int(item['row_index'])}: "
                f"instability={float(item['instability_score']):.4f}, "
                f"std={float(item['probability_std']):.4f}, "
                f"disagree={float(item['disagreement_rate']):.4f}, "
                f"mean_p={float(item['mean_probability']):.4f}, "
                f"flags={flags}",
            )
    elif kind == "prototype_audit":
        state.latest_prototype_audit_report = result
        _log(window, format_prototype_audit_summary(result))
        for item in result.get("prototypes", [])[:4]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  prototype row={int(item['row_index'])}: "
                f"label={int(item['label'])}, score={float(item['prototype_score']):.4f}, "
                f"opp_frac={float(item['local_opposite_fraction']):.4f}, flags={flags}",
            )
        for item in result.get("boundary_rows", [])[:4]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  boundary row={int(item['row_index'])}: "
                f"label={int(item['label'])}, score={float(item['boundary_score']):.4f}, "
                f"opp_frac={float(item['local_opposite_fraction']):.4f}, flags={flags}",
            )
    elif kind == "feature_separability":
        state.latest_feature_separability_report = result
        _log(window, format_feature_separability_summary(result))
        for item in result.get("features", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}: "
                f"AUC={float(item['auc']):.4f}, "
                f"bal_acc={float(item['best_balanced_accuracy']):.4f}, "
                f"SMD={float(item['standardized_mean_difference']):.4f}, "
                f"flags={flags}",
            )
        for item in result.get("redundant_pairs", [])[:3]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  redundant x{int(item['left_feature_index']) + 1}/x{int(item['right_feature_index']) + 1}: "
                f"corr={float(item['correlation']):.4f}, flags={flags}",
            )
    elif kind == "neighborhood_hardness":
        state.latest_neighborhood_hardness_report = result
        _log(window, format_neighborhood_hardness_summary(result))
        for item in result.get("rows", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  row={int(item['row_index'])}: "
                f"label={int(item['label'])}, vote={int(item['predicted_label'])}, "
                f"hardness={float(item['hardness_score']):.4f}, "
                f"opp_vote={float(item['opposite_vote_rate']):.4f}, flags={flags}",
            )
    elif kind == "dataset_triage":
        state.latest_dataset_triage_report = result
        state.latest_feature_separability_report = result.get("feature_separability")
        state.latest_prototype_audit_report = result.get("prototype_audit")
        state.latest_neighborhood_hardness_report = result.get("neighborhood_hardness")
        state.latest_ood_sentinel_report = result.get("ood_sentinel")
        _log(window, format_dataset_triage_summary(result))
        for action in result.get("summary", {}).get("top_actions", [])[:5]:
            _log(window, f"  action: {action}")
        if isinstance(state.latest_feature_separability_report, dict):
            _log(window, "  " + format_feature_separability_summary(state.latest_feature_separability_report))
        if isinstance(state.latest_prototype_audit_report, dict):
            _log(window, "  " + format_prototype_audit_summary(state.latest_prototype_audit_report))
        if isinstance(state.latest_neighborhood_hardness_report, dict):
            _log(window, "  " + format_neighborhood_hardness_summary(state.latest_neighborhood_hardness_report))
        if isinstance(state.latest_ood_sentinel_report, dict):
            _log(window, "  " + format_ood_sentinel_summary(state.latest_ood_sentinel_report))
    elif kind == "experiment_advisor":
        state.latest_experiment_advisor_report = result
        _log(window, format_experiment_advisor_summary(result))
        for item in result.get("recommendations", [])[:6]:
            _log(
                window,
                f"  {int(item.get('rank', 0))}. "
                f"[{item.get('priority', '-')}/{item.get('category', '-')}] "
                f"{item.get('title', '-')}: {item.get('action', '-')}",
            )
    elif kind == "trial_inspector":
        state.latest_trial_inspector_report = result
        _log(window, format_trial_inspector_summary(result))
        for item in result.get("leaderboard", [])[:5]:
            _log(
                window,
                f"  #{int(item.get('rank', 0))} trial={item.get('trial_index', '-')} "
                f"{item.get('backend', '-')}/{item.get('feature_map', '-')} "
                f"F1={float(item.get('f1') or 0.0):.4f} "
                f"acc={float(item.get('accuracy') or 0.0):.4f} "
                f"loss={_format_optional_metric(item.get('validation_loss'))}",
            )
        for item in result.get("recommendations", [])[:4]:
            _log(
                window,
                f"  next {int(item.get('rank', 0))}. "
                f"[{item.get('priority', '-')}/{item.get('category', '-')}] "
                f"{item.get('title', '-')}: {item.get('action', '-')}",
            )
    elif kind == "mps_sweep":
        state.latest_mps_sweep_report = result
        _log(window, format_mps_sweep_summary(result))
        for row in result.get("results", []):
            _log(
                window,
                f"  chi={int(row['bond_dim'])}: F1={float(row['f1']):.4f}, "
                f"Brier={float(row['brier_score']):.4f}, ECE={float(row['ece']):.4f}",
            )
    elif kind == "sample_review":
        state.latest_sample_review_report = result
        _log(window, format_sample_review_summary(result))
        for label, items in (
            ("label issue", result.get("label_issues", [])),
            ("hard", result.get("hard_examples", [])),
            ("ambiguous", result.get("ambiguous_examples", [])),
        ):
            for item in items[:3]:
                _log(
                    window,
                    f"  {label} row={int(item['row_index'])}: "
                    f"label={int(item['label'])}, pred={int(item['predicted_label'])}, "
                    f"p={float(item['probability']):.4f}, loss={float(item['loss']):.4f}",
                )
    elif kind == "permutation_null":
        state.latest_permutation_null_report = result
        _log(window, format_permutation_null_summary(result))
        observed = result.get("observed", {})
        p_values = result.get("p_values", {})
        for metric in ("f1", "accuracy", "balanced_accuracy"):
            distribution = result.get("null_distribution", {}).get(metric, {})
            _log(
                window,
                f"  {metric}: observed={float(observed.get(metric, 0.0)):.4f}, "
                f"null_mean={float(distribution.get('mean', 0.0)):.4f}, "
                f"p={float(p_values.get(metric, 1.0)):.4f}, "
                f"p95={float(distribution.get('p95', 0.0)):.4f}",
            )
    elif kind == "population_drift":
        state.latest_population_drift_report = result
        _log(window, format_population_drift_summary(result))
        for item in result.get("features", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}: "
                f"PSI={float(item['psi']):.4f}, "
                f"KS={float(item['ks_statistic']):.4f}, "
                f"mean_shift={float(item['mean_shift_std']):.4f}, "
                f"outside={float(item['outside_reference_rate']):.4f}, "
                f"flags={flags}",
            )
    elif kind == "adversarial_validation":
        state.latest_adversarial_validation_report = result
        _log(window, format_adversarial_validation_summary(result))
        for item in result.get("features", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}: "
                f"auc_drop={float(item['auc_drop']):.4f}, "
                f"acc_drop={float(item['accuracy_drop']):.4f}, "
                f"prob_shift={float(item['mean_probability_shift']):.4f}, "
                f"flags={flags}",
            )
    elif kind == "chronological_holdout":
        state.latest_chronological_holdout_report = result
        _log(window, format_chronological_holdout_summary(result))
        summary = result.get("summary", {})
        deltas = result.get("metric_deltas", {})
        _log(
            window,
            "  current replay: "
            f"F1_delta={float(deltas.get('f1_delta', 0.0)):.4f}, "
            f"acc_delta={float(deltas.get('accuracy_delta', 0.0)):.4f}, "
            f"Brier_delta={float(deltas.get('brier_score_delta', 0.0)):.4f}, "
            f"warning={summary.get('warning') or 'none'}",
        )
        current_baseline = result.get("current_baseline", {})
        if current_baseline.get("available"):
            baseline_metrics = current_baseline.get("current_model_metrics", {})
            baseline_deltas = current_baseline.get("metric_deltas_vs_reference_model", {})
            _log(
                window,
                "  current-only baseline: "
                f"F1={float(baseline_metrics.get('f1', 0.0)):.4f}, "
                f"F1_gain_vs_reference_model={float(baseline_deltas.get('f1_delta', 0.0)):.4f}",
            )
        else:
            _log(window, f"  current-only baseline unavailable: {current_baseline.get('reason', '-')}")
        for item in result.get("permutation_reliance", [])[:6]:
            flags = ",".join(item.get("risk_flags", [])) or "none"
            _log(
                window,
                f"  x{int(item['feature_index']) + 1}: "
                f"current_F1_drop={float(item['f1_drop']):.4f}, "
                f"logloss_increase={float(item['log_loss_increase']):.4f}, "
                f"prob_shift={float(item['mean_probability_shift']):.4f}, "
                f"flags={flags}",
            )
    elif kind == "multi_backend":
        best = select_best_from_runs(result)
        state.model = best.model
        state.latest_config = best.config
        state.latest_metrics = best.metrics
        state.latest_threshold = best.threshold
        state.preprocessor = best.preprocessor
        state.feature_importances = best.feature_importances
        state.trial_history = [_summarize_trial(item) for item in result]
        state.uncertainty_metadata = best.uncertainty
        state.latest_ablation_report = None
        state.latest_decision_curve_report = None
        state.latest_conformal_set_report = None
        state.latest_calibration_repair_report = None
        state.latest_selective_risk_report = None
        state.latest_sample_review_report = None
        state.latest_threshold_report = None
        state.latest_model_response_report = None
        state.latest_pairwise_interaction_report = None
        state.latest_slice_report = None
        state.latest_subgroup_disparity_report = None
        state.latest_stress_report = None
        state.latest_permutation_null_report = None
        state.latest_population_drift_report = None
        state.latest_adversarial_validation_report = None
        state.latest_chronological_holdout_report = None
        state.latest_cartography_report = None
        state.latest_ood_sentinel_report = None
        state.latest_bootstrap_stability_report = None
        state.latest_experiment_advisor_report = None
        state.latest_trial_inspector_report = None
        state.latest_mps_sweep_report = None
        for item in result:
            slot = ModelSlot(
                model=item.model,
                config=item.config,
                metrics=item.metrics.copy(),
                preprocessor=item.preprocessor,
                threshold=item.threshold,
                name=f"{item.config.backend} (F1: {item.metrics.get('f1', 0.0):.4f})",
            )
            state.model_slots.append(slot)
        state.active_slot_index = len(state.model_slots) - 1
        _update_slots_listbox(window, state)
        _log(window, f"Multi-backend sweep complete ({len(result)} runs). Best: {_format_config(best.config)}")
        _log(window, f"Best metrics: {_format_metrics(best.metrics)}")
    elif kind == "distill":
        training_result = result
        state.model = training_result.model
        state.latest_config = training_result.config
        state.latest_metrics = training_result.metrics
        state.latest_threshold = training_result.threshold
        state.preprocessor = training_result.preprocessor
        state.feature_importances = training_result.feature_importances
        state.trial_history = [_summarize_trial(training_result)]
        state.uncertainty_metadata = training_result.uncertainty
        state.latest_ablation_report = None
        state.latest_decision_curve_report = None
        state.latest_conformal_set_report = None
        state.latest_calibration_repair_report = None
        state.latest_selective_risk_report = None
        state.latest_sample_review_report = None
        state.latest_threshold_report = None
        state.latest_model_response_report = None
        state.latest_pairwise_interaction_report = None
        state.latest_slice_report = None
        state.latest_subgroup_disparity_report = None
        state.latest_stress_report = None
        state.latest_permutation_null_report = None
        state.latest_population_drift_report = None
        state.latest_adversarial_validation_report = None
        state.latest_chronological_holdout_report = None
        state.latest_cartography_report = None
        state.latest_ood_sentinel_report = None
        state.latest_bootstrap_stability_report = None
        state.latest_experiment_advisor_report = None
        state.latest_trial_inspector_report = None
        state.latest_mps_sweep_report = None
        
        # Auto-store in slots
        slot = ModelSlot(
            model=state.model,
            config=state.latest_config,
            metrics=state.latest_metrics.copy(),
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
            name=f"Distilled Student (F1: {state.latest_metrics.get('f1', 0.0):.4f})",
        )
        state.model_slots.append(slot)
        state.active_slot_index = len(state.model_slots) - 1
        _update_slots_listbox(window, state)
        
        _log(window, f"Distillation complete! Distilled student stored in slot '{slot.name}'.")
        _log(window, f"Metrics: {_format_metrics(training_result.metrics)}")
        _log(window, _format_calibration(training_result.metrics))
        _log(window, _format_uncertainty(training_result.uncertainty))


def _start_worker(window, state: AppState, status: str, task: Callable[[], tuple[str, Any]]) -> None:
    state.busy = True
    state.status_message = status
    _set_busy(window, True)
    window["-STATUS-"].update(status)
    _log(window, status)

    def run() -> None:
        try:
            window.write_event_value("-WORKER_DONE-", task())
        except Exception as exc:  # GUI thread renders the error.
            window.write_event_value("-WORKER_ERROR-", str(exc))

    threading.Thread(target=run, daemon=True).start()


def _replace_dataset(state: AppState, dataset, *, window=None) -> None:
    state.features = dataset.features.astype(float).tolist()
    state.labels = dataset.labels.astype(int).tolist()
    state.input_dim = dataset.input_dim
    _invalidate_model_artifacts(state)
    if window is not None:
        _refresh_audit_summary(window, state)


def _refresh_audit_summary(window, state: AppState) -> None:
    if len(state.labels) < 1:
        window["-AUDIT_SUMMARY-"].update("-")
        return
    summary = format_audit_summary(audit_dataset(state.features, state.labels))
    window["-AUDIT_SUMMARY-"].update(summary[:120] + ("..." if len(summary) > 120 else ""))


def _apply_preset_metadata(window, metadata: dict[str, Any]) -> None:
    defaults = metadata.get("training_defaults")
    if isinstance(defaults, dict):
        if defaults.get("epochs") is not None:
            window["-EPOCHS-"].update(str(defaults["epochs"]))
        if defaults.get("batch_size") is not None:
            window["-BATCH_SIZE-"].update(str(defaults["batch_size"]))
        if defaults.get("trials") is not None:
            window["-TRIALS-"].update(str(defaults["trials"]))
        if defaults.get("feature_map") in {"linear", "quadratic", "rff"}:
            window["-FEATURE_MAP-"].update(defaults["feature_map"])
        if defaults.get("l1_penalty") is not None:
            window["-L1_PENALTY-"].update(str(defaults["l1_penalty"]))
        if defaults.get("feature_selection_k") is not None:
            window["-FEATURE_K-"].update(str(defaults["feature_selection_k"]))
        if defaults.get("backend") is not None:
            window["-BACKEND-"].update(str(defaults["backend"]))
        if defaults.get("lr_schedule") is not None:
            window["-LR_SCHEDULE-"].update(str(defaults["lr_schedule"]))
        if defaults.get("gradient_clip") is not None:
            window["-GRADIENT_CLIP-"].update(str(defaults["gradient_clip"]))
        if defaults.get("mps_bond_dim") is not None:
            window["-MPS_BOND-"].update(str(defaults["mps_bond_dim"]))
        if defaults.get("mps_physical_dim") is not None:
            window["-MPS_PHYS-"].update(str(defaults["mps_physical_dim"]))
    recommended_map = metadata.get("recommended_feature_map")
    if recommended_map in {"linear", "quadratic", "rff"}:
        window["-FEATURE_MAP-"].update(recommended_map)


def _invalidate_model_artifacts(state: AppState) -> None:
    state.model = None
    state.latest_config = None
    state.latest_metrics = {}
    state.latest_threshold = 0.5
    state.preprocessor = None
    state.feature_importances = []
    state.trial_history = []
    state.uncertainty_metadata = {}
    state.latest_ablation_report = None
    state.latest_decision_curve_report = None
    state.latest_conformal_set_report = None
    state.latest_calibration_repair_report = None
    state.latest_selective_risk_report = None
    state.latest_sample_review_report = None
    state.latest_threshold_report = None
    state.latest_model_response_report = None
    state.latest_pairwise_interaction_report = None
    state.latest_slice_report = None
    state.latest_subgroup_disparity_report = None
    state.latest_stress_report = None
    state.latest_permutation_null_report = None
    state.latest_population_drift_report = None
    state.latest_adversarial_validation_report = None
    state.latest_chronological_holdout_report = None
    state.latest_cartography_report = None
    state.latest_ood_sentinel_report = None
    state.latest_bootstrap_stability_report = None
    state.latest_prototype_audit_report = None
    state.latest_feature_separability_report = None
    state.latest_neighborhood_hardness_report = None
    state.latest_dataset_triage_report = None
    state.latest_experiment_advisor_report = None
    state.latest_trial_inspector_report = None
    state.latest_mps_sweep_report = None


def _refresh_state(window, state: AppState) -> None:
    window["-SAMPLE_COUNT-"].update(str(len(state.labels)))
    window["-INPUT_DIM-"].update(str(state.input_dim) if state.input_dim is not None else "-")
    window["-DATASET_SUMMARY-"].update(_dataset_summary(state))
    window["-METRICS_SUMMARY-"].update(_metrics_summary(state))
    window["-STATUS-"].update(state.status_message if state.busy else "Ready")
    _update_slots_listbox(window, state)


def _set_busy(window, busy: bool) -> None:
    for key in (
        "-ADD_SAMPLE-",
        "-LOAD_BUILTIN_PRESET-",
        "-IMPORT_PRESET-",
        "-SAVE_PRESET-",
        "-LOAD_CSV-",
        "-SAVE_DATASET-",
        "-LOAD_DATASET-",
        "-CLEAR_DATA-",
        "-TRAIN_ONCE-",
        "-AUTO_EXPERIMENTS-",
        "-SAVE_MODEL-",
        "-LOAD_MODEL-",
        "-EXPORT_REPORT-",
        "-PREDICT-",
        "-COUNTERFACTUAL-",
        "-EXPORT_BATCH_PREDICTIONS-",
        "-IMPORT_REVIEWED_LABELS-",
        "-STORE_MODEL_SLOT-",
        "-ACTIVATE_MODEL_SLOT-",
        "-BUILD_ENSEMBLE-",
        "-COMPARE_MODELS-",
        "-WEIGHT_ANALYSIS-",
        "-DISTILL_MODEL-",
        "-MERGE_SLOTS-",
        "-RUN_MULTI_BACKEND-",
        "-PANEL_PREDICT-",
        "-SAVE_REGISTRY-",
        "-LOAD_REGISTRY-",
        "-BUILD_STACKED_ENSEMBLE-",
        "-AUDIT_DATASET-",
        "-POPULATION_DRIFT-",
        "-ADVERSARIAL_VALIDATION-",
        "-CHRONOLOGICAL_HOLDOUT-",
        "-LEARNING_CURVE-",
        "-ABLATION_DIAGNOSTICS-",
        "-STRESS_TEST-",
        "-SLICE_DIAGNOSTICS-",
        "-MODEL_RESPONSE-",
        "-PAIRWISE_INTERACTIONS-",
        "-SUBGROUP_DISPARITY-",
        "-THRESHOLD_DIAGNOSTICS-",
        "-DECISION_CURVE-",
        "-CONFORMAL_SETS-",
        "-CALIBRATION_REPAIR-",
        "-SELECTIVE_RISK-",
        "-SAMPLE_REVIEW-",
        "-PERMUTATION_NULL-",
        "-CARTOGRAPHY-",
        "-DATASET_TRIAGE-",
        "-EXPERIMENT_ADVISOR-",
        "-TRIAL_INSPECTOR-",
        "-OOD_SENTINEL-",
        "-BOOTSTRAP_STABILITY-",
        "-PROTOTYPE_AUDIT-",
        "-FEATURE_SEPARABILITY-",
        "-NEIGHBORHOOD_HARDNESS-",
        "-RELIABILITY-",
        "-MPS_BOND_SWEEP-",
        "-EXPORT_TRIALS-",
        "-SLOT_SIMILARITY-",
        "-SHAP_ANALYSIS-",
        "-DECISION_BOUNDARY-",
    ):
        if key in window.AllKeysDict:
            window[key].update(disabled=busy)


def _ensure_not_busy(state: AppState) -> None:
    if state.busy:
        raise ValueError("A training job is already running.")


def _required_path(path: str, label: str) -> str:
    if not path or not path.strip():
        raise ValueError(f"Choose a {label}.")
    return path.strip()


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _positive_int(raw_value: str, label: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label.capitalize()} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{label.capitalize()} must be a positive integer.")
    return value


def _nonnegative_float(raw_value: str, label: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label.capitalize()} must be a non-negative number.") from exc
    if value < 0.0:
        raise ValueError(f"{label.capitalize()} must be a non-negative number.")
    return value


def _int_value(raw_value: str, label: str) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label.capitalize()} must be an integer.") from exc


def _format_config(config: ModelConfig) -> str:
    parts = [
        f"layers={list(config.hidden_layers)}",
        f"lr={config.learning_rate:g}",
        f"batch={config.batch_size}",
        f"epochs={config.max_epochs}",
        f"map={config.feature_map}",
        f"backend={config.backend}",
    ]
    if getattr(config, 'l1_penalty', 0.0) > 0:
        parts.append(f"L1={config.l1_penalty:g}")
    if getattr(config, 'feature_selection_k', None) is not None:
        parts.append(f"feat_k={config.feature_selection_k}")
    return ", ".join(parts)


def _format_preset_metadata(metadata: dict[str, Any]) -> str:
    parts = []
    recommended = metadata.get("recommended_feature_map")
    if recommended:
        parts.append(f"recommended map={recommended}")
    examples = metadata.get("prediction_examples")
    if isinstance(examples, list) and examples:
        first = examples[0]
        parts.append(f"example {first.get('name', 'sample')}={first.get('features')}")
    return "Preset metadata: " + ", ".join(parts) if parts else "Preset metadata: none."


def _summarize_trial(result: ExperimentResult) -> dict[str, Any]:
    return {
        "config": result.config.to_dict(),
        "metrics": result.metrics,
        "threshold": result.threshold,
        "history": result.history,
        "feature_importances": result.feature_importances[:5],
        "uncertainty": result.uncertainty,
    }


def _dataset_summary(state: AppState) -> str:
    if not state.labels:
        return "No data"
    zeros = state.labels.count(0)
    ones = state.labels.count(1)
    trainable = "trainable" if zeros >= 2 and ones >= 2 else "needs 2 samples per class"
    return f"class 0={zeros}, class 1={ones}, {trainable}"


def _metrics_summary(state: AppState) -> str:
    if state.model is None or not state.latest_metrics:
        return "not trained"
    metrics = state.latest_metrics
    backend = getattr(state.latest_config, "backend", "?") if state.latest_config else "?"
    parts = [
        f"F1={float(metrics.get('f1', 0)):.3f}",
        f"acc={float(metrics.get('accuracy', 0)):.3f}",
    ]
    if "ece" in metrics:
        parts.append(f"ECE={float(metrics['ece']):.3f}")
    if "brier_score" in metrics:
        parts.append(f"Brier={float(metrics['brier_score']):.3f}")
    return f"{backend}: " + ", ".join(parts)


def _format_metrics(metrics: dict[str, float | int]) -> str:
    ordered = [
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "threshold",
        "fixed_threshold_f1",
        "fixed_threshold_accuracy",
        "threshold_gain_f1",
        "validation_loss",
        "brier_score",
        "ece",
        "conformal_coverage",
        "conformal_singleton_rate",
        "conformal_empty_rate",
        "class_weight_0",
        "class_weight_1",
        "bootstrap_f1_lower",
        "bootstrap_f1_upper",
        "bootstrap_accuracy_lower",
        "bootstrap_accuracy_upper",
        "bootstrap_balanced_accuracy_lower",
        "bootstrap_balanced_accuracy_upper",
    ]
    parts = []
    for key in ordered:
        if key in metrics:
            value = metrics[key]
            parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    for key in ("true_positive", "true_negative", "false_positive", "false_negative"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]}")
    # Cross-validation metrics
    cv_keys = sorted(k for k in metrics if k.startswith("cv_"))
    if cv_keys:
        parts.append("--- CV ---")
        for key in cv_keys:
            value = metrics[key]
            parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts) if parts else "no metrics"


def _format_optional_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _format_importances(importances: list[dict[str, float | int]]) -> str:
    if not importances:
        return "Top features: not available."
    parts = [
        f"f{item['feature_index']}={float(item['importance']):.4f}"
        for item in importances[:5]
    ]
    return "Top features: " + ", ".join(parts)


def _format_calibration(metrics: dict[str, float | int]) -> str:
    brier = metrics.get("brier_score")
    ece = metrics.get("ece")
    if brier is None and ece is None:
        return ""
    parts = ["Calibration diagnostics:"]
    if brier is not None:
        parts.append(f"Brier={float(brier):.4f}")
    if ece is not None:
        parts.append(f"ECE={float(ece):.4f}")
    return " ".join(parts)


def _format_uncertainty(uncertainty: dict[str, Any]) -> str:
    if not uncertainty:
        return ""
    parts = ["Uncertainty:"]
    source = uncertainty.get("conformal_source")
    if source:
        parts.append(f"source={source}")
    for key in (
        "conformal_alpha",
        "conformal_quantile",
        "conformal_coverage",
        "conformal_target_coverage",
        "conformal_singleton_rate",
        "conformal_empty_rate",
        "aps_alpha",
        "aps_tau",
        "aps_coverage",
        "aps_singleton_rate",
        "aps_mean_set_size",
    ):
        if key in uncertainty:
            parts.append(f"{key}={float(uncertainty[key]):.4f}")
    aps_source = uncertainty.get("aps_source")
    if aps_source:
        parts.append(f"aps_source={aps_source}")
    bootstrap_ci = uncertainty.get("bootstrap_ci")
    if bootstrap_ci:
        parts.append("bootstrap_ci={")
        ci_parts = []
        for metric, (lower, upper) in bootstrap_ci.items():
            ci_parts.append(f"{metric}=[{lower:.4f}, {upper:.4f}]")
        parts.append(", ".join(ci_parts) + "}")
    return " ".join(parts)


def _format_uncertainty_prediction(probability: float, uncertainty: dict[str, Any]) -> str:
    quantile = uncertainty.get("conformal_quantile") if uncertainty else None
    if quantile is None:
        return "."
    label_set = conformal_label_set(probability, float(quantile))
    if not label_set:
        label_text = "abstain"
    elif len(label_set) == 2:
        label_text = "{0,1}"
    else:
        label_text = "{" + str(label_set[0]) + "}"
    return f", conformal_set={label_text}, q={float(quantile):.4f}."


def _format_cv_summary(metrics: dict[str, float | int]) -> str:
    cv_folds = metrics.get("cv_folds")
    if cv_folds is None:
        return ""
    lines = [f"Cross-validation ({cv_folds} folds):"]
    for base_key in ["f1", "accuracy", "balanced_accuracy", "validation_loss", "brier_score", "ece"]:
        mean_key = f"cv_mean_{base_key}"
        std_key = f"cv_std_{base_key}"
        if mean_key in metrics:
            mean_val = float(metrics[mean_key])
            std_val = float(metrics.get(std_key, 0.0))
            lines.append(f"  {base_key}: {mean_val:.4f} +/- {std_val:.4f}")
    return "\n".join(lines)


def _log(window, message: str) -> None:
    if message:
        window["-LOG-"].update(f"{message}\n", append=True)


def _update_slots_listbox(window, state: AppState) -> None:
    if "-MODEL_SLOTS-" not in window.AllKeysDict:
        return
    display_list = []
    for idx, slot in enumerate(state.model_slots):
        prefix = "* " if idx == state.active_slot_index else "  "
        f1 = slot.metrics.get("f1", 0.0)
        display_list.append(f"{prefix}{slot.name} (F1: {f1:.4f})")
    window["-MODEL_SLOTS-"].update(values=display_list)


def _store_model_slot(window, state: AppState, values: dict[str, Any]) -> None:
    import PySimpleGUI as sg
    if state.model is None:
        _log(window, "No model trained yet to store.")
        return
    name = sg.popup_get_text("Enter a name for this model slot:", title="Store Model Slot", default_text=f"Model {len(state.model_slots) + 1}")
    if not name:
        return
    name = name.strip()
    if not name:
        return
    slot = ModelSlot(
        model=state.model,
        config=state.latest_config,
        metrics=state.latest_metrics.copy(),
        preprocessor=state.preprocessor,
        threshold=state.latest_threshold,
        name=name,
    )
    state.model_slots.append(slot)
    state.active_slot_index = len(state.model_slots) - 1
    _log(window, f"Stored model in slot '{name}' (F1: {slot.metrics.get('f1', 0.0):.4f}).")
    _update_slots_listbox(window, state)


def _activate_model_slot(window, state: AppState, values: dict[str, Any]) -> None:
    selected = values.get("-MODEL_SLOTS-")
    if not selected:
        _log(window, "Please select a slot from the listbox first.")
        return
    display_list = window["-MODEL_SLOTS-"].get_list_values()
    idx = display_list.index(selected[0])
    state.active_slot_index = idx
    slot = state.model_slots[idx]
    state.model = slot.model
    state.latest_config = slot.config
    state.latest_metrics = slot.metrics.copy()
    state.preprocessor = slot.preprocessor
    state.latest_threshold = slot.threshold
    state.latest_ablation_report = None
    state.latest_decision_curve_report = None
    state.latest_conformal_set_report = None
    state.latest_calibration_repair_report = None
    state.latest_selective_risk_report = None
    state.latest_sample_review_report = None
    state.latest_threshold_report = None
    state.latest_model_response_report = None
    state.latest_pairwise_interaction_report = None
    state.latest_slice_report = None
    state.latest_subgroup_disparity_report = None
    state.latest_stress_report = None
    state.latest_permutation_null_report = None
    state.latest_population_drift_report = None
    state.latest_adversarial_validation_report = None
    state.latest_chronological_holdout_report = None
    state.latest_cartography_report = None
    state.latest_ood_sentinel_report = None
    state.latest_bootstrap_stability_report = None
    state.latest_prototype_audit_report = None
    state.latest_feature_separability_report = None
    state.latest_neighborhood_hardness_report = None
    state.latest_experiment_advisor_report = None
    state.latest_trial_inspector_report = None
    state.latest_mps_sweep_report = None
    _log(window, f"Activated slot '{slot.name}'. Predictions and weight analysis will now run on this model.")
    _update_slots_listbox(window, state)


def _start_multi_backend_run(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)
    config = _config_from_values(values)
    queue = ModelRunQueue.multi_backend_sweep(config)

    def task() -> tuple[str, list[ExperimentResult]]:
        results = run_model_queue(
            dataset.features,
            dataset.labels,
            queue,
            progress_callback=lambda index, total, result: window.write_event_value(
                "-TRIAL_DONE-", (index, total, result)
            ),
        )
        return "multi_backend", results

    _start_worker(window, state, f"Running {len(queue.specs)} backend(s)...", task)


def _panel_predict(window, state: AppState, values: dict[str, Any]) -> None:
    if not state.model_slots:
        raise ValueError("Store at least one model slot before panel prediction.")
    vector_raw = values.get("-PREDICTION_VECTOR-", "").strip()
    if not vector_raw:
        raise ValueError("Enter a prediction vector JSON for panel deliberation.")
    vector = parse_prediction_vector(vector_raw, state.input_dim)
    fusion = values.get("-PANEL_FUSION-", state.panel_fusion)
    state.panel_fusion = fusion

    members = [
        PanelMember(
            name=slot.name,
            model=slot.model,
            preprocessor=slot.preprocessor,
            weight=max(1e-6, float(slot.metrics.get("f1", 1.0))),
            threshold=slot.threshold,
        )
        for slot in state.model_slots
    ]
    panel = ModelPanel(members, fusion=fusion)
    prediction = panel.predict(vector, threshold=state.latest_threshold)
    state.communication_log.extend(message.to_dict() for message in prediction.messages)
    _log(window, "=== Panel Deliberation ===")
    _log(window, panel.format_deliberation(prediction))
    consensus_label = 1 if float(prediction.consensus[0]) >= state.latest_threshold else 0
    _log(
        window,
        f"Panel outcome: label={consensus_label}, p={float(prediction.consensus[0]):.4f}, "
        f"disagreement={float(prediction.disagreement[0]):.4f}",
    )


def _save_registry(window, state: AppState, values: dict[str, Any]) -> None:
    if not state.model_slots:
        raise ValueError("No model slots to save.")
    path = save_model_registry(
        _required_path(values["-REGISTRY_PATH-"], "registry path"),
        state.model_slots,
        input_dim=state.input_dim,
    )
    _log(window, f"Saved {len(state.model_slots)} slot(s) to {path}.")


def _load_registry(window, state: AppState, values: dict[str, Any]) -> None:
    slots, input_dim = load_model_registry(_required_path(values["-REGISTRY_PATH-"], "registry path"))
    if state.input_dim is not None and input_dim is not None and input_dim != state.input_dim:
        raise ValueError(f"Registry expects {input_dim} features, current dataset uses {state.input_dim}.")
    state.model_slots = slots
    state.active_slot_index = 0 if slots else None
    if input_dim is not None and state.input_dim is None:
        state.input_dim = input_dim
    if slots:
        active = slots[state.active_slot_index or 0]
        state.model = active.model
        state.latest_config = active.config
        state.latest_metrics = active.metrics.copy()
        state.preprocessor = active.preprocessor
        state.latest_threshold = active.threshold
        state.latest_ablation_report = None
        state.latest_decision_curve_report = None
        state.latest_conformal_set_report = None
        state.latest_calibration_repair_report = None
        state.latest_selective_risk_report = None
        state.latest_sample_review_report = None
        state.latest_threshold_report = None
        state.latest_model_response_report = None
        state.latest_pairwise_interaction_report = None
        state.latest_slice_report = None
        state.latest_subgroup_disparity_report = None
        state.latest_stress_report = None
        state.latest_permutation_null_report = None
        state.latest_population_drift_report = None
        state.latest_adversarial_validation_report = None
        state.latest_chronological_holdout_report = None
        state.latest_cartography_report = None
        state.latest_ood_sentinel_report = None
        state.latest_bootstrap_stability_report = None
        state.latest_prototype_audit_report = None
        state.latest_feature_separability_report = None
        state.latest_neighborhood_hardness_report = None
        state.latest_experiment_advisor_report = None
        state.latest_trial_inspector_report = None
        state.latest_mps_sweep_report = None
    _update_slots_listbox(window, state)
    _log(window, f"Loaded {len(slots)} slot(s) from registry.")


def _build_ensemble(window, state: AppState, values: dict[str, Any]) -> None:
    if not state.model_slots:
        _log(window, "No model slots stored to build an ensemble.")
        return
    
    from .experiments import EnsemblePredictor, evaluate_predictions
    
    fusion = values.get("-PANEL_FUSION-", "mean")
    weights = [max(1e-6, float(slot.metrics.get("f1", 1.0))) for slot in state.model_slots] if fusion == "weighted" else None
    models = [(slot.model, slot.preprocessor) for slot in state.model_slots]
    ensemble = EnsemblePredictor(models, fusion=fusion, member_weights=weights)
    
    state.model = ensemble
    state.preprocessor = None
    state.latest_ablation_report = None
    state.latest_decision_curve_report = None
    state.latest_conformal_set_report = None
    state.latest_calibration_repair_report = None
    state.latest_selective_risk_report = None
    state.latest_sample_review_report = None
    state.latest_threshold_report = None
    state.latest_model_response_report = None
    state.latest_pairwise_interaction_report = None
    state.latest_slice_report = None
    state.latest_subgroup_disparity_report = None
    state.latest_stress_report = None
    state.latest_permutation_null_report = None
    state.latest_population_drift_report = None
    state.latest_adversarial_validation_report = None
    state.latest_chronological_holdout_report = None
    state.latest_cartography_report = None
    state.latest_ood_sentinel_report = None
    state.latest_bootstrap_stability_report = None
    state.latest_prototype_audit_report = None
    state.latest_feature_separability_report = None
    state.latest_neighborhood_hardness_report = None
    state.latest_experiment_advisor_report = None
    state.latest_trial_inspector_report = None
    state.latest_mps_sweep_report = None
    
    state.latest_config = ModelConfig(
        lr_schedule="constant",
        gradient_clip=0.0,
    )
    
    if state.features and state.labels:
        dataset = validate_dataset(state.features, state.labels, min_samples=4)
        probs = ensemble.predict(dataset.features).reshape(-1)
        metrics = evaluate_predictions(dataset.labels, probs, threshold=0.5)
        state.latest_metrics = metrics
        _log(window, f"Built Ensemble with {len(models)} models. Evaluated on full dataset (F1: {metrics.get('f1', 0.0):.4f}).")
    else:
        state.latest_metrics = {}
        _log(window, f"Built Ensemble with {len(models)} models (no evaluation dataset available).")
        
    slot = ModelSlot(
        model=ensemble,
        config=state.latest_config,
        metrics=state.latest_metrics.copy(),
        preprocessor=None,
        threshold=0.5,
        name=f"Ensemble [{fusion}] ({len(models)} models)",
    )
    state.model_slots.append(slot)
    state.active_slot_index = len(state.model_slots) - 1
    _update_slots_listbox(window, state)


def _build_stacked_ensemble(window, state: AppState, values: dict[str, Any]) -> None:
    if len(state.model_slots) < 2:
        _log(window, "Need at least 2 model slots for stacked fusion.")
        return
    if not state.features or not state.labels:
        _log(window, "Load a dataset to fit stacking weights on validation data.")
        return

    from .experiments import EnsemblePredictor, evaluate_predictions, split_train_validation

    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)
    x_train, y_train, x_val, y_val = split_train_validation(dataset, seed=42)
    members = [
        PanelMember(
            name=slot.name,
            model=slot.model,
            preprocessor=slot.preprocessor,
            threshold=slot.threshold,
        )
        for slot in state.model_slots
    ]
    stacking_coef = fit_stacking_weights(members, x_val, y_val)
    models = [(slot.model, slot.preprocessor) for slot in state.model_slots]
    ensemble = EnsemblePredictor(models, fusion="stacking", stacking_coef=stacking_coef)
    state.model = ensemble
    state.preprocessor = None
    state.latest_config = ModelConfig(backend="auto", feature_map=values.get("-FEATURE_MAP-", "linear"))
    state.latest_ablation_report = None
    state.latest_decision_curve_report = None
    state.latest_conformal_set_report = None
    state.latest_calibration_repair_report = None
    state.latest_selective_risk_report = None
    state.latest_sample_review_report = None
    state.latest_threshold_report = None
    state.latest_model_response_report = None
    state.latest_pairwise_interaction_report = None
    state.latest_slice_report = None
    state.latest_subgroup_disparity_report = None
    state.latest_stress_report = None
    state.latest_permutation_null_report = None
    state.latest_population_drift_report = None
    state.latest_adversarial_validation_report = None
    state.latest_chronological_holdout_report = None
    state.latest_cartography_report = None
    state.latest_ood_sentinel_report = None
    state.latest_bootstrap_stability_report = None
    state.latest_prototype_audit_report = None
    state.latest_feature_separability_report = None
    state.latest_neighborhood_hardness_report = None
    state.latest_experiment_advisor_report = None
    state.latest_trial_inspector_report = None
    state.latest_mps_sweep_report = None
    probs = ensemble.predict(dataset.features).reshape(-1)
    metrics = evaluate_predictions(dataset.labels, probs, threshold=0.5)
    state.latest_metrics = metrics
    slot = ModelSlot(
        model=ensemble,
        config=state.latest_config,
        metrics=metrics.copy(),
        preprocessor=None,
        threshold=0.5,
        name=f"Stacked Ensemble ({len(models)} models)",
    )
    state.model_slots.append(slot)
    state.active_slot_index = len(state.model_slots) - 1
    _update_slots_listbox(window, state)
    _log(window, f"Built stacked ensemble with {len(models)} members (F1: {metrics.get('f1', 0.0):.4f}).")


def _compare_models(window, state: AppState, values: dict[str, Any]) -> None:
    if not state.model_slots:
        _log(window, "No model slots stored to compare.")
        return
    
    _log(window, "=== Model Comparison ===")
    header = f"{'Model Name':<25} | {'F1':<8} | {'Accuracy':<8} | {'Brier':<8} | {'ECE':<8}"
    _log(window, header)
    _log(window, "-" * len(header))
    for slot in state.model_slots:
        f1 = slot.metrics.get("f1", 0.0)
        acc = slot.metrics.get("accuracy", 0.0)
        brier = slot.metrics.get("brier_score", 0.0)
        ece = slot.metrics.get("ece", 0.0)
        _log(window, f"{slot.name:<25} | {f1:<8.4f} | {acc:<8.4f} | {brier:<8.4f} | {ece:<8.4f}")
    _log(window, "========================")



def _start_cartography(window, state: AppState) -> None:
    _ensure_not_busy(state)
    if state.model is None:
        raise ValueError("Train or load a model before running dataset cartography.")
    dataset = validate_dataset(state.features, state.labels, min_samples=1, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_dataset_cartography(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "cartography", report

    _start_worker(window, state, "Running dataset cartography...", task)


def _start_ood_sentinel(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=False)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_ood_sentinel(
            state.model,
            dataset.features,
            dataset.labels,
            preprocessor=state.preprocessor,
            threshold=state.latest_threshold,
        )
        return "ood_sentinel", report

    mode = "model-aware" if state.model is not None else "model-free"
    _start_worker(window, state, f"Running OOD sentinel ({mode})...", task)


def _start_bootstrap_stability(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=8, require_two_classes=True)
    feature_map = values.get("-FEATURE_MAP-", "linear")
    if feature_map not in {"linear", "quadratic", "rff"}:
        feature_map = "linear"
    try:
        max_epochs = min(max(5, int(values.get("-EPOCHS-", 45))), 90)
    except (TypeError, ValueError):
        max_epochs = 45

    def task() -> tuple[str, dict[str, Any]]:
        report = run_bootstrap_stability_diagnostics(
            dataset.features,
            dataset.labels,
            max_epochs=max_epochs,
            feature_map=feature_map,
            threshold=state.latest_threshold,
        )
        return "bootstrap_stability", report

    _start_worker(window, state, "Running bootstrap stability committee...", task)


def _start_prototype_audit(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=6, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_prototype_audit(dataset.features, dataset.labels)
        return "prototype_audit", report

    _start_worker(window, state, "Running nearest-neighbor prototype audit...", task)


def _start_feature_separability(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=6, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_feature_separability_diagnostics(dataset.features, dataset.labels)
        return "feature_separability", report

    _start_worker(window, state, "Running feature separability lens...", task)


def _start_neighborhood_hardness(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=6, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_neighborhood_hardness_diagnostics(dataset.features, dataset.labels)
        return "neighborhood_hardness", report

    _start_worker(window, state, "Running neighborhood hardness scan...", task)


def _start_dataset_triage(window, state: AppState) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=6, require_two_classes=True)

    def task() -> tuple[str, dict[str, Any]]:
        report = run_dataset_triage(dataset.features, dataset.labels)
        return "dataset_triage", report

    _start_worker(window, state, "Running dataset triage workflow...", task)


def _start_experiment_advisor(window, state: AppState) -> None:
    _ensure_not_busy(state)

    def task() -> tuple[str, dict[str, Any]]:
        report = build_experiment_advisor(
            sample_count=len(state.labels),
            input_dim=state.input_dim,
            labels=state.labels,
            config=state.latest_config,
            metrics=state.latest_metrics,
            trial_history=state.trial_history,
            dataset_triage_report=state.latest_dataset_triage_report,
            feature_separability_report=state.latest_feature_separability_report,
            neighborhood_hardness_report=state.latest_neighborhood_hardness_report,
            prototype_audit_report=state.latest_prototype_audit_report,
            ood_sentinel_report=state.latest_ood_sentinel_report,
            threshold_report=state.latest_threshold_report,
            calibration_repair_report=state.latest_calibration_repair_report,
            decision_curve_report=state.latest_decision_curve_report,
            selective_risk_report=state.latest_selective_risk_report,
            stress_report=state.latest_stress_report,
            permutation_null_report=state.latest_permutation_null_report,
            population_drift_report=state.latest_population_drift_report,
            adversarial_validation_report=state.latest_adversarial_validation_report,
            chronological_holdout_report=state.latest_chronological_holdout_report,
        )
        return "experiment_advisor", report

    _start_worker(window, state, "Building next-experiment advisor...", task)


def _start_trial_inspector(window, state: AppState) -> None:
    _ensure_not_busy(state)

    def task() -> tuple[str, dict[str, Any]]:
        report = inspect_trial_history(state.trial_history)
        return "trial_inspector", report

    _start_worker(window, state, "Inspecting trial history...", task)


def _run_reliability_diagram(window, state: AppState) -> None:
    if state.model is None:
        raise ValueError("Train or load a model before running a reliability diagram.")
    from .analysis import format_reliability_summary
    from .modeling import predict_probability

    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=False)
    prepared = state.preprocessor.transform(dataset.features) if state.preprocessor else dataset.features
    probabilities = predict_probability(state.model, prepared).reshape(-1)
    _log(window, format_reliability_summary(dataset.labels, probabilities))


def _start_mps_bond_sweep(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=8, require_two_classes=True)
    config = _config_from_values(values)
    config = ModelConfig.from_dict({**config.to_dict(), "backend": "mps"})

    def task() -> tuple[str, dict[str, Any]]:
        report = run_mps_bond_sweep(dataset.features, dataset.labels, config)
        return "mps_sweep", report

    _start_worker(window, state, "Running MPS bond-dimension sweep...", task)


def _export_trials(window, state: AppState, values: dict[str, Any]) -> None:
    if not state.trial_history:
        raise ValueError("No trial history to export. Train once or run auto experiments first.")
    path = export_trial_history_csv(_required_path(values["-TRIAL_CSV_PATH-"], "trial CSV path"), state.trial_history)
    window["-TRIAL_CSV_PATH-"].update(str(path))
    _log(window, f"Exported {len(state.trial_history)} trial(s) to {path}.")


def _run_slot_similarity(window, state: AppState) -> None:
    if len(state.model_slots) < 2:
        raise ValueError("Store at least two models in the registry for similarity analysis.")
    from .analysis import format_similarity_matrix, registry_similarity_matrix

    report = registry_similarity_matrix(state.model_slots)
    for line in format_similarity_matrix(report).splitlines():
        _log(window, line)


def _run_weight_analysis(window, state: AppState, values: dict[str, Any]) -> None:
    if state.model is None:
        _log(window, "No active model to analyze.")
        return
    
    try:
        from .analysis import weight_statistics
        stats = weight_statistics(state.model)
        
        _log(window, f"=== Weight Analysis ({state.model.__class__.__name__}) ===")
        _log(window, f"  Mean:      {stats['mean']:.6f}")
        _log(window, f"  Std Dev:   {stats['std']:.6f}")
        _log(window, f"  Min:       {stats['min']:.6f}")
        _log(window, f"  Max:       {stats['max']:.6f}")
        _log(window, f"  Sparsity:  {stats['sparsity']:.2f}% zeros")
        _log(window, f"  L1 Norm:   {stats['l1_norm']:.6f}")
        _log(window, f"  L2 Norm:   {stats['l2_norm']:.6f}")
        _log(window, "==========================")
    except Exception as exc:
        _log(window, f"Weight analysis error: {exc}")


def _run_shap_analysis(window, state: AppState, values: dict[str, Any]) -> None:
    if state.model is None:
        _log(window, "No active model to analyze. Train or load a model first.")
        return
    raw = values.get("-PREDICTION_VECTOR-", "").strip()
    if not raw:
        _log(window, "Please enter a prediction vector JSON first.")
        return
    try:
        from .data import parse_prediction_vector
        import numpy as np
        sample_list = parse_prediction_vector(raw, state.input_dim)
        sample = np.array(sample_list, dtype=np.float32)
        from .explainability import compute_shap_attributions, render_shap_bar_chart
        attributions = compute_shap_attributions(state.model, sample, state.preprocessor)
        chart = render_shap_bar_chart(attributions)
        _log(window, "\n" + chart)
    except Exception as exc:
        _log(window, f"SHAP analysis error: {exc}")


def _run_decision_boundary(window, state: AppState, values: dict[str, Any]) -> None:
    if state.model is None:
        _log(window, "No active model. Train or load a model first.")
        return
    if not state.features:
        _log(window, "No dataset loaded.")
        return
    try:
        import numpy as np
        x = np.array(state.features, dtype=np.float32)
        y = np.array(state.labels, dtype=np.int32)
        from .explainability import generate_decision_boundary_map
        grid_map = generate_decision_boundary_map(state.model, x, y, state.preprocessor)
        _log(window, "\n" + grid_map)
    except Exception as exc:
        _log(window, f"Decision boundary visualization error: {exc}")


def _distill_model(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    import PySimpleGUI as sg
    
    selected = values.get("-MODEL_SLOTS-")
    if not selected:
        if state.model is None:
            _log(window, "Please select a teacher model slot from the registry first, or train a model.")
            return
        teacher_model = state.model
        teacher_prep = state.preprocessor
        teacher_name = "Active Model"
    else:
        display_list = window["-MODEL_SLOTS-"].get_list_values()
        idx = display_list.index(selected[0])
        slot = state.model_slots[idx]
        teacher_model = slot.model
        teacher_prep = slot.preprocessor
        teacher_name = slot.name

    alpha_str = sg.popup_get_text("Enter distillation coefficient alpha (0.0 to 1.0):", title="Knowledge Distillation", default_text="0.5")
    if not alpha_str:
        return
    try:
        alpha = float(alpha_str)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError()
    except ValueError:
        raise ValueError("Distillation alpha must be a float between 0.0 and 1.0.")

    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)

    config = _config_from_values(values)

    from .experiments import train_distilled_model

    def task() -> tuple[str, ExperimentResult]:
        res = train_distilled_model(dataset.features, dataset.labels, teacher_model, teacher_prep, config, alpha)
        return "distill", res

    _start_worker(window, state, f"Distilling student from {teacher_name}...", task)


def _merge_slots(window, state: AppState, values: dict[str, Any]) -> None:
    if not state.model_slots:
        _log(window, "No model slots stored to merge.")
        return

    numpy_slots = [slot for slot in state.model_slots if isinstance(slot.model, NumpyBinaryClassifier)]
    if len(numpy_slots) < 2:
        _log(window, "Need at least 2 NumPy model slots stored in the registry to perform merging.")
        return

    try:
        from .experiments import merge_models, evaluate_predictions
        models_list = [(slot.model, slot.preprocessor) for slot in numpy_slots]

        import PySimpleGUI as sg
        choice = sg.popup_yes_no("Weight models by their validation F1 scores? (No = equal weights)", title="Merge Weights")
        if choice == "Yes":
            weights = [max(1e-4, slot.metrics.get("f1", 0.0)) for slot in numpy_slots]
            _log(window, f"Merging {len(numpy_slots)} models weighted by F1: {weights}")
        else:
            weights = None
            _log(window, f"Merging {len(numpy_slots)} models with equal weights.")

        merged_model, merged_prep = merge_models(models_list, weights)

        state.model = merged_model
        state.preprocessor = merged_prep
        state.latest_ablation_report = None
        state.latest_decision_curve_report = None
        state.latest_conformal_set_report = None
        state.latest_calibration_repair_report = None
        state.latest_selective_risk_report = None
        state.latest_sample_review_report = None
        state.latest_threshold_report = None
        state.latest_model_response_report = None
        state.latest_pairwise_interaction_report = None
        state.latest_slice_report = None
        state.latest_subgroup_disparity_report = None
        state.latest_stress_report = None
        state.latest_permutation_null_report = None
        state.latest_population_drift_report = None
        state.latest_adversarial_validation_report = None
        state.latest_chronological_holdout_report = None
        state.latest_cartography_report = None
        state.latest_ood_sentinel_report = None
        state.latest_bootstrap_stability_report = None
        state.latest_prototype_audit_report = None
        state.latest_feature_separability_report = None
        state.latest_neighborhood_hardness_report = None
        state.latest_experiment_advisor_report = None
        state.latest_trial_inspector_report = None
        state.latest_mps_sweep_report = None
        state.latest_config = ModelConfig(
            lr_schedule="constant",
            gradient_clip=0.0,
        )

        if state.features and state.labels:
            dataset = validate_dataset(state.features, state.labels, min_samples=4)
            if merged_prep is not None:
                x_std = merged_prep.transform(dataset.features)
            else:
                x_std = dataset.features
            probs = merged_model.predict(x_std).reshape(-1)
            metrics = evaluate_predictions(dataset.labels, probs, threshold=0.5)
            state.latest_metrics = metrics
            state.latest_threshold = 0.5
            _log(window, f"Merged model evaluated on full dataset (F1: {metrics.get('f1', 0.0):.4f}).")
        else:
            state.latest_metrics = {}
            state.latest_threshold = 0.5
            _log(window, "Merged model created (no evaluation dataset available).")

        slot = ModelSlot(
            model=merged_model,
            config=state.latest_config,
            metrics=state.latest_metrics.copy(),
            preprocessor=merged_prep,
            threshold=state.latest_threshold,
            name=f"Merged Slot (F1: {state.latest_metrics.get('f1', 0.0):.4f})",
        )
        state.model_slots.append(slot)
        state.active_slot_index = len(state.model_slots) - 1
        _update_slots_listbox(window, state)

    except Exception as exc:
        _log(window, f"Merge error: {exc}")
