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
from .modeling import ModelConfig, predict_probability
from .persistence import load_dataset, load_model_bundle, save_dataset, save_model_bundle
from .preprocessing import FeatureStandardizer
from .presets import generate_builtin_preset, load_preset_file, preset_labels, preset_metadata, save_preset_file
from .reporting import build_experiment_report, export_experiment_report
from .scoring import score_prediction_csv


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
        [
            sg.Text("Epochs"),
            sg.Input("50", key="-EPOCHS-", size=(6, 1)),
            sg.Text("Batch"),
            sg.Input("16", key="-BATCH_SIZE-", size=(6, 1)),
            sg.Text("Trials"),
            sg.Input("8", key="-TRIALS-", size=(6, 1)),
            sg.Text("Map"),
            sg.Combo(["linear", "quadratic", "rff"], default_value="rff", readonly=True, key="-FEATURE_MAP-", size=(10, 1)),
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
        [sg.Button("Train once", key="-TRAIN_ONCE-"), sg.Button("Run auto experiments", key="-AUTO_EXPERIMENTS-")],
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

    return [
        [sg.Column(data_column, expand_x=True), sg.VSeparator(), sg.Column(training_column, expand_x=True)],
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
    _replace_dataset(state, dataset)
    metadata = preset_metadata(values["-PRESET_NAME-"])
    _apply_preset_metadata(window, metadata)
    _log(window, f"Loaded preset '{values['-PRESET_NAME-']}' with {dataset.sample_count} samples.")
    _log(window, _format_preset_metadata(metadata))


def _import_preset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset, metadata = load_preset_file(_required_path(values["-PRESET_PATH-"], "preset path"))
    _replace_dataset(state, dataset)
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
    _replace_dataset(state, dataset)
    _log(window, f"Loaded {dataset.sample_count} samples from CSV.")


def _save_dataset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = validate_dataset(state.features, state.labels, min_samples=1)
    path = save_dataset(_required_path(values["-DATASET_PATH-"], "dataset path"), dataset)
    _log(window, f"Saved dataset to {path}.")


def _load_dataset(window, state: AppState, values: dict[str, Any]) -> None:
    dataset = load_dataset(_required_path(values["-DATASET_PATH-"], "dataset path"))
    _replace_dataset(state, dataset)
    _log(window, f"Loaded {dataset.sample_count} samples from dataset JSON.")


def _clear_data(window, state: AppState) -> None:
    state.features.clear()
    state.labels.clear()
    state.input_dim = None
    _invalidate_model_artifacts(state)
    _log(window, "Cleared dataset.")


def _start_train_once(window, state: AppState, values: dict[str, Any]) -> None:
    _ensure_not_busy(state)
    dataset = validate_dataset(state.features, state.labels, min_samples=4, require_two_classes=True)

    # Parse L1 penalty
    l1_raw = values.get("-L1_PENALTY-", "0.0").strip()
    l1_penalty = float(l1_raw) if l1_raw else 0.0

    # Parse feature selection k
    feat_k_raw = values.get("-FEATURE_K-", "").strip()
    feature_selection_k = int(feat_k_raw) if feat_k_raw else None

    config = ModelConfig(
        hidden_layers=(32,),
        learning_rate=0.001,
        batch_size=_positive_int(values["-BATCH_SIZE-"], "batch size"),
        max_epochs=_positive_int(values["-EPOCHS-"], "epochs"),
        feature_map=values["-FEATURE_MAP-"],
        l1_penalty=l1_penalty,
        feature_selection_k=feature_selection_k,
    )

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


def _replace_dataset(state: AppState, dataset) -> None:
    state.features = dataset.features.astype(float).tolist()
    state.labels = dataset.labels.astype(int).tolist()
    state.input_dim = dataset.input_dim
    _invalidate_model_artifacts(state)


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
    ):
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
    ):
        if key in uncertainty:
            parts.append(f"{key}={float(uncertainty[key]):.4f}")
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
            lines.append(f"  {base_key}: {mean_val:.4f} ± {std_val:.4f}")
    return "\n".join(lines)


def _log(window, message: str) -> None:
    if message:
        window["-LOG-"].update(f"{message}\n", append=True)
