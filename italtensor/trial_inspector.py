from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Mapping, Sequence


def inspect_trial_history(
    trial_history: Sequence[Mapping[str, Any]] | None,
    *,
    top_n: int = 6,
) -> dict[str, Any]:
    """Summarize trial evidence and recommend the next bounded search step."""
    raw_trials = list(trial_history or [])
    top_n = max(1, int(top_n))
    rows: list[dict[str, Any]] = []
    invalid_count = 0
    for index, trial in enumerate(raw_trials, start=1):
        row = _trial_row(index, trial)
        if row is None:
            invalid_count += 1
        else:
            rows.append(row)

    if not rows:
        warning = "No valid trial metrics were found." if raw_trials else None
        recommendations = [
            _recommendation(
                1,
                "high",
                "search",
                "Create a trial history",
                "Trial-history inspection needs at least one Train once, auto-experiment, or multi-backend run.",
                "Run Train once for a baseline, then run auto experiments with 8-16 trials before comparing settings.",
                "trial_history",
            )
        ]
        return {
            "trial_count": len(raw_trials),
            "valid_trial_count": 0,
            "invalid_trial_count": invalid_count,
            "summary": {
                "best_trial_index": None,
                "best_rank": None,
                "best_f1": None,
                "best_accuracy": None,
                "best_validation_loss": None,
                "best_backend": None,
                "best_feature_map": None,
                "leader_margin_f1": None,
                "winner_stability": None,
                "search_diversity": _empty_diversity(),
                "recommendation": recommendations[0]["action"],
                "warning": warning,
            },
            "leaderboard": [],
            "groups": [],
            "recommendations": recommendations,
        }

    ranked = sorted(rows, key=_rank_key, reverse=True)
    leaderboard = [_public_row(row, rank) for rank, row in enumerate(ranked[:top_n], start=1)]
    groups = _group_leaderboard(rows)
    diversity = _diversity(rows)
    recommendations = _build_recommendations(ranked, groups, diversity)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    best_group = _best_group_for_trial(groups, best)
    leader_margin = None if runner_up is None else best["f1"] - runner_up["f1"]
    stability = None
    if best_group is not None:
        stability = {
            "group": best_group["group"],
            "group_count": best_group["count"],
            "group_share": _round(best_group["count"] / max(len(rows), 1)),
            "group_avg_f1": best_group["avg_f1"],
        }
    warning = _warning(rows, invalid_count)
    return {
        "trial_count": len(raw_trials),
        "valid_trial_count": len(rows),
        "invalid_trial_count": invalid_count,
        "summary": {
            "best_trial_index": best["trial_index"],
            "best_rank": 1,
            "best_f1": _round(best["f1"]),
            "best_accuracy": _round(best["accuracy"]),
            "best_validation_loss": _round(best["validation_loss"]),
            "best_backend": best["backend"],
            "best_feature_map": best["feature_map"],
            "leader_margin_f1": _round(leader_margin),
            "winner_stability": stability,
            "search_diversity": diversity,
            "recommendation": recommendations[0]["action"] if recommendations else None,
            "warning": warning,
        },
        "leaderboard": leaderboard,
        "groups": groups[:top_n],
        "recommendations": recommendations,
    }


def format_trial_inspector_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    best_f1 = summary.get("best_f1")
    best_f1_text = "-" if best_f1 is None else f"{float(best_f1):.4f}"
    margin = summary.get("leader_margin_f1")
    margin_text = "-" if margin is None else f"{float(margin):.4f}"
    return (
        "Trial inspector: "
        f"valid_trials={int(report.get('valid_trial_count', 0))}, "
        f"best=trial {summary.get('best_trial_index') or '-'} "
        f"{summary.get('best_backend') or '-'}/{summary.get('best_feature_map') or '-'} "
        f"F1={best_f1_text}, "
        f"margin={margin_text}, "
        f"next={summary.get('recommendation') or 'none'}"
    )


def _trial_row(index: int, trial: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(trial, Mapping):
        return None
    config = trial.get("config")
    metrics = trial.get("metrics")
    if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
        return None
    f1 = _metric(metrics, "f1", "val_f1", default=0.0)
    accuracy = _metric(metrics, "accuracy", "val_accuracy", "balanced_accuracy", default=0.0)
    validation_loss = _metric(metrics, "validation_loss", "val_loss", "log_loss", "loss", default=float("inf"))
    if not isfinite(f1) or not isfinite(accuracy):
        return None
    return {
        "trial_index": index,
        "backend": _text(config.get("backend"), "auto"),
        "feature_map": _text(config.get("feature_map"), "linear"),
        "hidden_layers": _hidden_layers(config.get("hidden_layers")),
        "learning_rate": _optional_float(config.get("learning_rate")),
        "batch_size": _optional_int(config.get("batch_size")),
        "max_epochs": _optional_int(config.get("max_epochs")),
        "threshold": _optional_float(trial.get("threshold", metrics.get("threshold"))),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": _optional_float(metrics.get("balanced_accuracy")),
        "validation_loss": float(validation_loss),
        "brier_score": _optional_float(metrics.get("brier_score")),
        "ece": _optional_float(metrics.get("ece")),
        "roc_auc": _optional_float(metrics.get("roc_auc")),
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    loss = row["validation_loss"]
    loss_score = -loss if isfinite(loss) else float("-inf")
    return (row["f1"], row["accuracy"], loss_score, -int(row["trial_index"]))


def _public_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "trial_index": row["trial_index"],
        "backend": row["backend"],
        "feature_map": row["feature_map"],
        "hidden_layers": row["hidden_layers"],
        "learning_rate": _round(row["learning_rate"]),
        "batch_size": row["batch_size"],
        "max_epochs": row["max_epochs"],
        "threshold": _round(row["threshold"]),
        "f1": _round(row["f1"]),
        "accuracy": _round(row["accuracy"]),
        "balanced_accuracy": _round(row["balanced_accuracy"]),
        "validation_loss": _round(row["validation_loss"]),
        "brier_score": _round(row["brier_score"]),
        "ece": _round(row["ece"]),
        "roc_auc": _round(row["roc_auc"]),
    }


def _group_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["feature_map"])].append(row)
    output: list[dict[str, Any]] = []
    for (backend, feature_map), items in groups.items():
        ranked = sorted(items, key=_rank_key, reverse=True)
        best = ranked[0]
        output.append(
            {
                "group": f"{backend}/{feature_map}",
                "backend": backend,
                "feature_map": feature_map,
                "count": len(items),
                "best_trial_index": best["trial_index"],
                "best_f1": _round(best["f1"]),
                "best_accuracy": _round(best["accuracy"]),
                "best_validation_loss": _round(best["validation_loss"]),
                "avg_f1": _round(_mean(item["f1"] for item in items)),
                "avg_accuracy": _round(_mean(item["accuracy"] for item in items)),
                "avg_validation_loss": _round(_mean(item["validation_loss"] for item in items if isfinite(item["validation_loss"]))),
                "hidden_layer_variants": sorted({_hidden_label(item["hidden_layers"]) for item in items}),
            }
        )
    return sorted(
        output,
        key=lambda item: (
            float(item.get("best_f1") or 0.0),
            float(item.get("avg_f1") or 0.0),
            int(item.get("count") or 0),
            str(item.get("group") or ""),
        ),
        reverse=True,
    )


def _best_group_for_trial(groups: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    group_name = f"{row['backend']}/{row['feature_map']}"
    for group in groups:
        if group.get("group") == group_name:
            return group
    return None


def _diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    backends = sorted({row["backend"] for row in rows})
    feature_maps = sorted({row["feature_map"] for row in rows})
    hidden_layers = sorted({_hidden_label(row["hidden_layers"]) for row in rows})
    learning_rates = sorted({_round(row["learning_rate"]) for row in rows if row["learning_rate"] is not None})
    batch_sizes = sorted({row["batch_size"] for row in rows if row["batch_size"] is not None})
    return {
        "backend_count": len(backends),
        "feature_map_count": len(feature_maps),
        "hidden_layer_count": len(hidden_layers),
        "learning_rate_count": len(learning_rates),
        "batch_size_count": len(batch_sizes),
        "unique_backends": backends,
        "unique_feature_maps": feature_maps,
        "unique_hidden_layers": hidden_layers,
        "unique_learning_rates": learning_rates,
        "unique_batch_sizes": batch_sizes,
    }


def _empty_diversity() -> dict[str, Any]:
    return {
        "backend_count": 0,
        "feature_map_count": 0,
        "hidden_layer_count": 0,
        "learning_rate_count": 0,
        "batch_size_count": 0,
        "unique_backends": [],
        "unique_feature_maps": [],
        "unique_hidden_layers": [],
        "unique_learning_rates": [],
        "unique_batch_sizes": [],
    }


def _build_recommendations(
    ranked: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    diversity: dict[str, Any],
) -> list[dict[str, Any]]:
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = None if second is None else best["f1"] - second["f1"]
    recs: list[dict[str, Any]] = []

    def add(
        score: float,
        priority: str,
        category: str,
        title: str,
        reason: str,
        action: str,
        source: str,
    ) -> None:
        recs.append(
            {
                "priority_score": float(score),
                "priority": priority,
                "category": category,
                "title": title,
                "reason": reason,
                "action": action,
                "source": source,
            }
        )

    if len(ranked) < 3:
        add(
            88.0,
            "high",
            "search",
            "Increase trial evidence",
            f"Only {len(ranked)} valid trial(s) are available, so the current winner may be noise.",
            "Run auto experiments with at least 8 trials before locking a model or backend.",
            "trial_count",
        )
    elif len(ranked) < 8:
        add(
            70.0,
            "medium",
            "search",
            "Complete a small random-search budget",
            f"{len(ranked)} valid trials exist; the default evidence budget is 8 or more.",
            "Run another bounded auto-experiment sweep until the trial history has at least 8 comparable runs.",
            "trial_count",
        )

    if diversity["backend_count"] < 2:
        add(
            62.0,
            "medium",
            "backend",
            "Compare at least two backends",
            "The history covers only one backend, so backend choice is not evidenced.",
            "Run Multi-Backend with the current best hyperparameters, then inspect this report again.",
            "search_diversity",
        )
    if diversity["feature_map_count"] < 2:
        add(
            60.0,
            "medium",
            "feature_map",
            "Probe a second feature map",
            "All trials use the same feature map, so nonlinear or linear alternatives have not been ruled out.",
            "Run auto experiments that include linear, quadratic, and RFF feature maps.",
            "search_diversity",
        )

    if best["f1"] < 0.60:
        add(
            84.0,
            "high",
            "model_selection",
            "Do not promote the current winner yet",
            f"The best validation F1 is {best['f1']:.3f}, below the practical 0.60 checkpoint.",
            "Run dataset triage, then expand the search space before saving the model as final.",
            "best_metrics",
        )
    elif margin is not None and margin <= 0.02:
        add(
            76.0,
            "high",
            "stability",
            "Resolve a narrow leaderboard margin",
            f"The best F1 margin over the runner-up is only {margin:.3f}.",
            "Rerun the top two settings with a different seed or cross-validation before choosing a winner.",
            "leader_margin",
        )

    best_brier = best.get("brier_score")
    best_ece = best.get("ece")
    if (best_brier is not None and best_brier >= 0.22) or (best_ece is not None and best_ece >= 0.08):
        add(
            58.0,
            "medium",
            "calibration",
            "Inspect probability calibration",
            f"Best trial calibration is Brier={_display_metric(best_brier)}, ECE={_display_metric(best_ece)}.",
            "Run Calibration repair and compare repaired Brier/ECE before using probabilities operationally.",
            "calibration_metrics",
        )

    if best["accuracy"] - best["f1"] >= 0.18:
        add(
            56.0,
            "medium",
            "imbalance",
            "Check threshold and class imbalance",
            f"Accuracy ({best['accuracy']:.3f}) is much higher than F1 ({best['f1']:.3f}).",
            "Run Threshold tradeoff and inspect class counts; optimize for F1, recall, or utility rather than accuracy alone.",
            "metric_gap",
        )

    if groups:
        top_group = groups[0]
        runner_group = groups[1] if len(groups) > 1 else None
        if (
            int(top_group.get("count", 0)) >= 2
            and runner_group is not None
            and float(top_group.get("avg_f1") or 0.0) >= float(runner_group.get("avg_f1") or 0.0) + 0.03
        ):
            add(
                54.0,
                "medium",
                "exploit",
                "Focus the next sweep around the leading family",
                f"{top_group['group']} has the strongest average F1 among repeated groups.",
                f"Keep {top_group['backend']}/{top_group['feature_map']} fixed next, then vary learning rate, regularization, and epochs.",
                "group_leaderboard",
            )

    if not recs:
        add(
            40.0,
            "low",
            "promotion",
            "Promote with an evidence note",
            "The trial history has no obvious thin-search, margin, diversity, calibration, or score warning.",
            "Save the model and exported report, including this trial inspector section as the model-selection note.",
            "trial_history",
        )

    ordered = sorted(recs, key=lambda item: (-item["priority_score"], item["category"], item["title"]))
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
    return ordered[:8]


def _warning(rows: list[dict[str, Any]], invalid_count: int) -> str | None:
    parts: list[str] = []
    if invalid_count:
        parts.append(f"{invalid_count} malformed trial record(s) were ignored")
    if len(rows) < 3:
        parts.append("trial evidence is very thin")
    if any(row["validation_loss"] == float("inf") for row in rows):
        parts.append("some trials have no comparable validation loss")
    return "; ".join(parts) if parts else None


def _recommendation(
    rank: int,
    priority: str,
    category: str,
    title: str,
    reason: str,
    action: str,
    source: str,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "priority_score": 100.0 - rank,
        "priority": priority,
        "category": category,
        "title": title,
        "reason": reason,
        "action": action,
        "source": source,
    }


def _metric(metrics: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value
    return default


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hidden_layers(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _hidden_label(hidden_layers: Sequence[int]) -> str:
    return "[" + ",".join(str(int(item)) for item in hidden_layers) + "]"


def _text(value: Any, default: str) -> str:
    text = str(value or default).strip()
    return text or default


def _mean(values: Any) -> float | None:
    collected = [float(value) for value in values]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return round(parsed, 6)


def _display_metric(value: Any) -> str:
    rounded = _round(value)
    return "-" if rounded is None else f"{rounded:.3f}"
