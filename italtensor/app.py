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
from .model_runner import ModelRunQueue, available_backends, run_model_queue, select_best_from_runs
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
from .scoring import score_prediction_csv
from .audit import audit_dataset, format_audit_summary
from .learning_curves import learning_curve_points


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
    window = sg.Window("Italtensor Premium Workbench", _layout(sg), finalize=True)
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
            elif event == "-LEARNING_CURVE-":
                _start_learning_curve(window, state, values)
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
            elif event == "-EXPORT_BATCH_PREDICTIONS-":
                _start_batch_predictions(window, state, values)
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
        [
            sg.Text("Samples:"),
            sg.Text("0", key="-SAMPLE_COUNT-", size=(6, 1)),
            sg.Text("Input dim:"),
            sg.Text("-", key="-INPUT_DIM-", size=(6, 1)),
        ],
        [sg.Text("Dataset:"), sg.Text("No data", key="-DATASET_SUMMARY-", expand_x=True)],
        [sg.Text("Audit:"), sg.Text("-", key="-AUDIT_SUMMARY-", expand_x=True)],
        [sg.Button("Audit dataset", key="-AUDIT_DATASET-"), sg.Button("Learning curve", key="-LEARNING_CURVE-")],
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
            sg.Button("Train once", key="-TRAIN_ONCE-"),
            sg.Button("Run auto experiments", key="-AUTO_EXPERIMENTS-"),
            sg.Button("Weight Analysis", key="-WEIGHT_ANALYSIS-"),
            sg.Text("MPS chi"),
            sg.Input("8", key="-MPS_BOND-", size=(4, 1)),
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
        [sg.Input(key="-PREDICTION_VECTOR-", expand_x=True), sg.Button("Predict", key="-PREDICT-")],
        [sg.Text("Batch prediction CSV")],
        [
            sg.Input(key="-BATCH_INPUT_PATH-", expand_x=True),
            sg.FileBrowse(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
        ],
        [
            sg.Input(key="-BATCH_OUTPUT_PATH-", expand_x=True),
            sg.FileSaveAs(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
            sg.Button("Export batch predictions", key="-EXPORT_BATCH_PREDICTIONS-"),
        ],
        [sg.Text("Status:"), sg.Text("Ready", key="-STATUS-", expand_x=True)],
    ]

    slots_column = [
        [sg.Text("Model Registry / Multi-Model")],
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

    return [
        [
            sg.Column(data_column, expand_x=True, vertical_alignment="top"),
            sg.VSeparator(),
            sg.Column(training_column, expand_x=True, vertical_alignment="top"),
            sg.VSeparator(),
            sg.Column(slots_column, expand_x=True, vertical_alignment="top"),
        ],
        [sg.Multiline(size=(110, 18), key="-LOG-", autoscroll=True, disabled=True, expand_x=True, expand_y=True)],
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
    )
    _log(window, f"Saved preset '{values['-PRESET_SAVE_NAME-']}' to {path}.")


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
    )


def _start_train_once(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)
    config = _config_from_values(values)

    use_cv = values.get("-USE_CV-", False)
    if use_cv:
        n_splits = _positive_int(values.get("-KFOLD_SPLITS-", "5"), "CV folds")
        def task() -> tuple[str, ExperimentResult]:
            return "single", train_single_model_cv(dataset.features, dataset.labels, config, n_splits=n_splits)
        _start_worker(window, state, f"Training with {n_splits}-Fold CV...", task)
    else:
        def task() -> tuple[str, ExperimentResult]:
            return "single", train_single_model(dataset.features, dataset.labels, config)
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
    _log(window, f"Loaded model expecting {state.input_dim} features.")


def _export_report(window, state: AppState, values: dict[str, Any]) -> None:
    if state.latest_config is None and not state.latest_metrics:
        raise ValueError("Train or load a model before exporting a report.")
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


def _refresh_state(window, state: AppState) -> None:
    window["-SAMPLE_COUNT-"].update(str(len(state.labels)))
    window["-INPUT_DIM-"].update(str(state.input_dim) if state.input_dim is not None else "-")
    window["-DATASET_SUMMARY-"].update(_dataset_summary(state))
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
        "-EXPORT_BATCH_PREDICTIONS-",
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
        "-LEARNING_CURVE-",
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


def _positive_int(raw_value: str, label: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label.capitalize()} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{label.capitalize()} must be a positive integer.")
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
