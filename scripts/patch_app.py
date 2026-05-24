"""One-shot patcher to update italtensor/app.py after restore from git."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "app_restore.py"
target = ROOT / "italtensor" / "app.py"
text = source.read_text(encoding="utf-8")

if "from . import __version__" not in text:
    text = text.replace(
        "from .thresholds import format_threshold_summary, run_threshold_diagnostics\n\n\n@dataclass",
        "from .thresholds import format_threshold_summary, run_threshold_diagnostics\n"
        "from .cartography import format_cartography_summary, run_dataset_cartography\n"
        "from .mps_diagnostics import format_mps_sweep_summary, run_mps_bond_sweep\n"
        "from .trials_io import export_trial_history_csv\n"
        "from . import __version__\n\n\n@dataclass",
    )

text = text.replace(
    "    latest_stress_report: dict[str, Any] | None = None\n    busy: bool = False",
    "    latest_stress_report: dict[str, Any] | None = None\n"
    "    latest_cartography_report: dict[str, Any] | None = None\n"
    "    latest_mps_sweep_report: dict[str, Any] | None = None\n    busy: bool = False",
)

text = text.replace(
    '    window = sg.Window("Italtensor Premium Workbench", _layout(sg), finalize=True)',
    '    window = sg.Window(f"Italtensor Workbench v{__version__}", _layout(sg), finalize=True, resizable=True)',
)

text = text.replace(
    """            elif event == "-SAMPLE_REVIEW-":
                _start_sample_review(window, state)
            elif event == "-TRAIN_ONCE-":""",
    """            elif event == "-SAMPLE_REVIEW-":
                _start_sample_review(window, state)
            elif event == "-CARTOGRAPHY-":
                _start_cartography(window, state)
            elif event == "-RELIABILITY-":
                _run_reliability_diagram(window, state)
            elif event == "-MPS_BOND_SWEEP-":
                _start_mps_bond_sweep(window, state, values)
            elif event == "-EXPORT_TRIALS-":
                _export_trials(window, state, values)
            elif event == "-SLOT_SIMILARITY-":
                _run_slot_similarity(window, state)
            elif event == "-TRAIN_ONCE-":""",
)

# layout tweaks
text = text.replace(
    '    data_column = [\n        [sg.Text("Training sample JSON")]',
    '    data_column = [\n        [sg.Text("Data", font=("Segoe UI", 11, "bold"))],\n        [sg.Text("Training sample JSON")]',
)

old_training = """    training_column = [
        [
            sg.Text("Samples:"),
            sg.Text("0", key="-SAMPLE_COUNT-", size=(6, 1)),
            sg.Text("Input dim:"),
            sg.Text("-", key="-INPUT_DIM-", size=(6, 1)),
        ],
        [sg.Text("Dataset:"), sg.Text("No data", key="-DATASET_SUMMARY-", expand_x=True)],
        [sg.Text("Audit:"), sg.Text("-", key="-AUDIT_SUMMARY-", expand_x=True)],
        [
            sg.Button("Audit dataset", key="-AUDIT_DATASET-"),
            sg.Button("Learning curve", key="-LEARNING_CURVE-"),
            sg.Button("Ablation diagnostics", key="-ABLATION_DIAGNOSTICS-"),
            sg.Button("Stress test", key="-STRESS_TEST-"),
            sg.Button("Slice diagnostics", key="-SLICE_DIAGNOSTICS-"),
            sg.Button("Threshold tradeoff", key="-THRESHOLD_DIAGNOSTICS-"),
            sg.Button("Sample review", key="-SAMPLE_REVIEW-"),
        ],"""

new_training = """    training_column = [
        [sg.Text("Train & diagnose", font=("Segoe UI", 11, "bold"))],
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
            sg.Button("Audit dataset", key="-AUDIT_DATASET-"),
            sg.Button("Learning curve", key="-LEARNING_CURVE-"),
            sg.Button("Ablation", key="-ABLATION_DIAGNOSTICS-"),
            sg.Button("Stress test", key="-STRESS_TEST-"),
        ],
        [
            sg.Button("Slice diagnostics", key="-SLICE_DIAGNOSTICS-"),
            sg.Button("Threshold tradeoff", key="-THRESHOLD_DIAGNOSTICS-"),
            sg.Button("Sample review", key="-SAMPLE_REVIEW-"),
            sg.Button("Cartography", key="-CARTOGRAPHY-"),
        ],
        [
            sg.Button("Reliability diagram", key="-RELIABILITY-"),
            sg.Button("MPS bond sweep", key="-MPS_BOND_SWEEP-"),
            sg.Button("Export trials CSV", key="-EXPORT_TRIALS-"),
        ],"""

if old_training in text:
    text = text.replace(old_training, new_training)

text = text.replace(
    """            sg.Text("MPS chi"),
            sg.Input("8", key="-MPS_BOND-", size=(4, 1)),
        ],
        [sg.Text("Model path")],""",
    """            sg.Text("MPS chi"),
            sg.Input("8", key="-MPS_BOND-", size=(4, 1)),
            sg.Text("phys"),
            sg.Input("4", key="-MPS_PHYS-", size=(4, 1)),
        ],
        [sg.Text("Trial CSV")],
        [
            sg.Input(key="-TRIAL_CSV_PATH-", expand_x=True),
            sg.FileSaveAs(file_types=(("CSV files", "*.csv"), ("All files", "*.*"))),
        ],
        [sg.Text("Model path")],""",
)

text = text.replace(
    '        [sg.Text("Model Registry / Multi-Model")],',
    '        [sg.Text("Registry & panel", font=("Segoe UI", 11, "bold"))],',
)

text = text.replace(
    """            sg.Button("Compare Models", key="-COMPARE_MODELS-", expand_x=True),
            sg.Button("Distill Model", key="-DISTILL_MODEL-", expand_x=True),
        ],
        [
            sg.Button("Merge Slots", key="-MERGE_SLOTS-", expand_x=True),""",
    """            sg.Button("Compare Models", key="-COMPARE_MODELS-", expand_x=True),
            sg.Button("Slot similarity", key="-SLOT_SIMILARITY-", expand_x=True),
        ],
        [
            sg.Button("Distill Model", key="-DISTILL_MODEL-", expand_x=True),
        ],
        [
            sg.Button("Merge Slots", key="-MERGE_SLOTS-", expand_x=True),""",
)

text = text.replace(
    '        [sg.Multiline(size=(110, 18), key="-LOG-", autoscroll=True, disabled=True, expand_x=True, expand_y=True)]',
    """        [
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
        ]""",
)

text = text.replace(
    """    if mps_bond_dim < 2:
        raise ValueError("MPS bond dimension must be at least 2.")
    return ModelConfig(""",
    """    if mps_bond_dim < 2:
        raise ValueError("MPS bond dimension must be at least 2.")
    mps_phys_raw = values.get("-MPS_PHYS-", "4").strip()
    try:
        mps_physical_dim = int(mps_phys_raw) if mps_phys_raw else 4
    except ValueError as exc:
        raise ValueError("MPS physical dimension must be an integer.") from exc
    if mps_physical_dim < 2:
        raise ValueError("MPS physical dimension must be at least 2.")
    return ModelConfig(""",
)

text = text.replace(
    "        mps_bond_dim=mps_bond_dim,\n    )",
    "        mps_bond_dim=mps_bond_dim,\n        mps_physical_dim=mps_physical_dim,\n    )",
)

text = text.replace(
    """    window["-DATASET_SUMMARY-"].update(_dataset_summary(state))
    window["-STATUS-"].update(state.status_message if state.busy else "Ready")""",
    """    window["-DATASET_SUMMARY-"].update(_dataset_summary(state))
    window["-METRICS_SUMMARY-"].update(_metrics_summary(state))
    window["-STATUS-"].update(state.status_message if state.busy else "Ready")""",
)

text = text.replace(
    '        "-SAMPLE_REVIEW-",\n    ):',
    '        "-SAMPLE_REVIEW-",\n        "-CARTOGRAPHY-",\n        "-RELIABILITY-",\n'
    '        "-MPS_BOND_SWEEP-",\n        "-EXPORT_TRIALS-",\n        "-SLOT_SIMILARITY-",\n    ):',
)

if "def _metrics_summary" not in text:
    text = text.replace(
        '    return f"class 0={zeros}, class 1={ones}, {trainable}"\n\n\ndef _format_metrics',
        '    return f"class 0={zeros}, class 1={ones}, {trainable}"\n\n\n'
        "def _metrics_summary(state: AppState) -> str:\n"
        "    if state.model is None or not state.latest_metrics:\n"
        '        return "not trained"\n'
        "    metrics = state.latest_metrics\n"
        '    backend = getattr(state.latest_config, "backend", "?") if state.latest_config else "?"\n'
        "    parts = [\n"
        '        f"F1={float(metrics.get(\'f1\', 0)):.3f}",\n'
        '        f"acc={float(metrics.get(\'accuracy\', 0)):.3f}",\n'
        "    ]\n"
        '    if "ece" in metrics:\n'
        '        parts.append(f"ECE={float(metrics[\'ece\']):.3f}")\n'
        '    if "brier_score" in metrics:\n'
        '        parts.append(f"Brier={float(metrics[\'brier_score\']):.3f}")\n'
        '    return f"{backend}: " + ", ".join(parts)\n\n\ndef _format_metrics',
    )

handlers = '''

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

'''

if "def _start_cartography" not in text:
    text = text.replace("\ndef _run_weight_analysis(window", handlers + "\ndef _run_weight_analysis(window")

worker_cases = '''
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
    elif kind == "mps_sweep":
        state.latest_mps_sweep_report = result
        _log(window, format_mps_sweep_summary(result))
        for row in result.get("results", []):
            _log(
                window,
                f"  chi={int(row['bond_dim'])}: F1={float(row['f1']):.4f}, "
                f"Brier={float(row['brier_score']):.4f}, ECE={float(row['ece']):.4f}",
            )
'''

if 'elif kind == "cartography"' not in text:
    text = text.replace('    elif kind == "sample_review":', worker_cases + '    elif kind == "sample_review":')

target.write_text(text, encoding="utf-8")
print(f"Wrote {target} ({len(text)} bytes)")
