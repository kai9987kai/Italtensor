from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .data import Dataset, validate_dataset
from .modeling import ModelConfig, train_model, predict_probability, NumpyBinaryClassifier
from .preprocessing import FeatureStandardizer


@dataclass
class ExperimentResult:
    config: ModelConfig
    metrics: dict[str, float | int]
    history: dict[str, list[float]]
    model: Any = None
    threshold: float = 0.5
    preprocessor: FeatureStandardizer | None = None
    feature_importances: list[dict[str, float | int]] = field(default_factory=list)


def split_train_validation(
    dataset: Dataset,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of dataset into training and validation sets.
    
    Requires at least two samples for each class.
    """
    x = dataset.features
    y = dataset.labels
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    if class_counts.get(0, 0) < 2 or class_counts.get(1, 0) < 2:
        raise ValueError("Dataset must have at least two samples for each class to split.")
        
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    
    for c in [0, 1]:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        # Stratified division
        n_train = max(1, min(n - 1, int(round(n * train_ratio))))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])
        
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    # Shuffle splits to remove ordered blocks
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def balanced_class_weights(labels: np.ndarray) -> dict[int, float] | None:
    """Calculate balanced class weights using inverse frequency."""
    y = np.asarray(labels, dtype=np.int32)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return None
        
    n_samples = len(y)
    n_classes = len(unique)
    weights = {}
    for c, count in zip(unique, counts):
        weights[int(c)] = float(n_samples / (n_classes * count))
    return weights


def compute_ece(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.asarray(labels, dtype=np.int32)
    y_prob = np.asarray(probabilities, dtype=np.float32)
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def fit_platt_scaling(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling (logistic calibration) on predicted probabilities and true labels."""
    p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(p / (1.0 - p))
    y = np.asarray(labels, dtype=np.float32)
    
    a, b = 1.0, 0.0
    lr = 0.1
    # Simple gradient descent to optimize log-likelihood
    for _ in range(100):
        preds = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        errors = preds - y
        grad_a = np.mean(errors * logits)
        grad_b = np.mean(errors)
        a -= lr * grad_a
        b -= lr * grad_b
        
    return float(a), float(b)


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate predictions against true labels using specified threshold."""
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("Labels and probabilities must be the same length.")
        
    y_pred = (y_prob >= threshold).astype(np.int32)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    accuracy = float((tp + tn) / len(y_true))
    
    rec_0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    rec_1 = recall
    balanced_accuracy = float((rec_0 + rec_1) / 2)
    
    # Calculate log loss
    probs_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    val_loss = float(-np.mean(y_true * np.log(probs_clipped) + (1.0 - y_true) * np.log(1.0 - probs_clipped)))
    
    # Calibration metrics
    brier_score = float(np.mean((y_prob - y_true) ** 2))
    ece = compute_ece(y_true, y_prob)
    
    return {
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "validation_loss": val_loss,
        "brier_score": brier_score,
        "ece": ece,
    }


def optimize_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on validation data."""
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    
    if len(y_true) == 0:
        return 0.5
        
    candidates = np.unique(y_prob)
    if len(candidates) == 0:
        return 0.5
        
    best_f1 = -1.0
    best_threshold = 0.5
    
    for thresh in candidates:
        metrics = evaluate_predictions(y_true, y_prob, thresh)
        f1 = metrics["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)
        elif abs(f1 - best_f1) < 1e-9:
            # Tie breaker: prefer threshold closer to 0.5
            if abs(thresh - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(thresh)
                
    return best_threshold


def generate_random_configs(trials: int = 24, seed: int = 7) -> list[ModelConfig]:
    """Generate deterministic random model configs."""
    configs = []
    for hl in [(16,), (32,), (64,), (64, 32)]:
        for lr in [0.01, 0.001, 0.0003]:
            for fm in ["linear", "quadratic", "rff"]:
                for bs in [8, 16, 32]:
                    for me in [25, 50]:
                        configs.append(ModelConfig(
                            hidden_layers=hl,
                            learning_rate=lr,
                            feature_map=fm,
                            batch_size=bs,
                            max_epochs=me,
                        ))
                        
    if trials > len(configs):
        raise ValueError(f"Number of trials cannot exceed {len(configs)}.")
        
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(configs), size=trials, replace=False)
    return [configs[i] for i in indices]


def select_best_result(results: list[ExperimentResult]) -> ExperimentResult:
    """Select the best experiment result based on F1, Accuracy, and Validation Loss."""
    if not results:
        raise ValueError("Cannot select best result from an empty list.")
        
    def ranking_key(res: ExperimentResult) -> tuple[float, float, float]:
        f1 = float(res.metrics.get("f1", 0.0))
        acc = float(res.metrics.get("accuracy", 0.0))
        val_loss = float(res.metrics.get("validation_loss", float("inf")))
        return (f1, acc, -val_loss)
        
    return max(results, key=ranking_key)


def permutation_feature_importance(
    model: Any,
    preprocessor: FeatureStandardizer | None,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    max_features: int = 5,
    repeats: int = 5,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    """Calculate feature importances using permutation method on validation data."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    n_samples, n_features = x.shape
    
    # Calculate baseline metrics
    if preprocessor is not None:
        x_std = preprocessor.transform(x)
    else:
        x_std = x.copy()
        
    probs = predict_probability(model, x_std)
    baseline_metrics = evaluate_predictions(y, probs, threshold)
    baseline_acc = baseline_metrics["accuracy"]
    
    rng = np.random.default_rng(seed)
    importances = []
    
    for j in range(n_features):
        scores = []
        for _ in range(repeats):
            x_perm = x.copy()
            # Permute the values of raw feature j
            shuffled_idx = rng.permutation(n_samples)
            x_perm[:, j] = x[shuffled_idx, j]
            
            if preprocessor is not None:
                x_perm_std = preprocessor.transform(x_perm)
            else:
                x_perm_std = x_perm
                
            perm_probs = predict_probability(model, x_perm_std)
            perm_metrics = evaluate_predictions(y, perm_probs, threshold)
            scores.append(perm_metrics["accuracy"])
            
        mean_acc = float(np.mean(scores))
        importance_val = float(baseline_acc - mean_acc)
        importances.append({
            "feature_index": j,
            "importance": importance_val,
        })
        
    importances.sort(key=lambda item: item["importance"], reverse=True)
    return importances[:max_features]


def train_single_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
) -> ExperimentResult:
    """Core single model training pipeline with post-hoc probability calibration."""
    # Validate and package input
    dataset = validate_dataset(features.tolist(), labels.tolist(), min_samples=4, require_two_classes=True)
    
    # Split train/validation
    x_train, y_train, x_val, y_val = split_train_validation(dataset)
    
    # Fit standardizer on train features only
    if getattr(config, "feature_selection_k", None) is not None:
        preprocessor = FeatureStandardizer.fit_with_selection(x_train, y_train, k=config.feature_selection_k)
    else:
        preprocessor = FeatureStandardizer.fit(x_train)
    x_train_std = preprocessor.transform(x_train)
    x_val_std = preprocessor.transform(x_val)
    
    # Compute balanced class weights
    class_weight = balanced_class_weights(y_train)
    
    # Train the model
    model, history = train_model(
        x_train_std,
        y_train,
        config,
        validation_data=(x_val_std, y_val),
        class_weight=class_weight,
    )
    
    # Perform post-hoc Platt scaling calibration if it's a Numpy model
    if isinstance(model, NumpyBinaryClassifier):
        # 1. Predict uncalibrated probabilities on validation
        uncal_val_probs = predict_probability(model, x_val_std)
        # 2. Fit Platt scaling
        a, b = fit_platt_scaling(uncal_val_probs, y_val)
        # 3. Apply calibration parameters to model
        model.calibration_a = a
        model.calibration_b = b
        
    # Get final calibrated validation probabilities
    val_probs = predict_probability(model, x_val_std)
    
    # Optimize threshold
    threshold = optimize_threshold(y_val, val_probs)
    
    # Evaluate metrics
    metrics = evaluate_predictions(y_val, val_probs, threshold)
    
    # Add validation loss and class weights
    if "val_loss" in history and history["val_loss"]:
        metrics["validation_loss"] = float(history["val_loss"][-1])
    else:
        probs_clipped = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
        metrics["validation_loss"] = float(-np.mean(y_val * np.log(probs_clipped) + (1.0 - y_val) * np.log(1.0 - probs_clipped)))
        
    metrics["threshold"] = threshold
    if class_weight is not None:
        metrics["class_weight_0"] = float(class_weight.get(0, 1.0))
        metrics["class_weight_1"] = float(class_weight.get(1, 1.0))
    else:
        metrics["class_weight_0"] = 1.0
        metrics["class_weight_1"] = 1.0
        
    # Permutation feature importance
    feature_importances = permutation_feature_importance(
        model,
        preprocessor,
        x_val,
        y_val,
        threshold=threshold,
        max_features=x_train.shape[1],
    )
    
    return ExperimentResult(
        config=config,
        metrics=metrics,
        history=history,
        model=model,
        threshold=threshold,
        preprocessor=preprocessor,
        feature_importances=feature_importances,
    )


def run_experiments(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    configs: list[ModelConfig] | None = None,
    trials: int = 8,
    trainer: Callable | None = None,
    progress_callback: Callable[[int, int, ExperimentResult], None] | None = None,
) -> list[ExperimentResult]:
    """Run multiple experiment configurations."""
    if configs is None:
        configs = generate_random_configs(trials=trials)
        
    if trainer is None:
        trainer = train_single_model
        
    results = []
    total = len(configs)
    
    for index, config in enumerate(configs, start=1):
        # Handle trainers expecting splits vs those expecting full dataset
        try:
            sig = inspect.signature(trainer)
            has_splits = len(sig.parameters) == 5
        except Exception:
            has_splits = False
            
        if has_splits:
            dataset = Dataset(features, labels, features.shape[1])
            x_train, y_train, x_val, y_val = split_train_validation(dataset)
            result = trainer(x_train, y_train, x_val, y_val, config)
        else:
            result = trainer(features, labels, config)
            
        results.append(result)
        if progress_callback is not None:
            progress_callback(index, total, result)
            
    return results


def stratified_kfold_indices(
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold train/validation indices."""
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Dataset must have at least two classes to perform stratified K-Fold.")
    for c, count in zip(unique, counts):
        if count < n_splits:
            raise ValueError(f"Class {c} has only {count} samples, which is less than the number of splits {n_splits}.")

    rng = np.random.default_rng(seed)
    class_indices = []
    for c in unique:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        class_indices.append(idx)

    folds = [[] for _ in range(n_splits)]
    for idx_list in class_indices:
        for i, idx in enumerate(idx_list):
            folds[i % n_splits].append(idx)

    folds_arr = [np.array(fold) for fold in folds]

    splits = []
    for val_fold_idx in range(n_splits):
        val_idx = folds_arr[val_fold_idx]
        train_idx = np.concatenate([folds_arr[i] for i in range(n_splits) if i != val_fold_idx])
        rng.shuffle(train_idx)
        splits.append((train_idx, val_idx))

    return splits


def train_single_model_cv(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig,
    n_splits: int = 5,
) -> ExperimentResult:
    """Train single model with K-Fold cross-validation, updating result metrics with CV stats."""
    dataset = validate_dataset(features.tolist(), labels.tolist(), min_samples=4, require_two_classes=True)
    x = dataset.features
    y = dataset.labels
    
    splits = stratified_kfold_indices(y, n_splits=n_splits, seed=config.random_seed)
    fold_metrics = []
    
    for train_idx, val_idx in splits:
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        
        if getattr(config, "feature_selection_k", None) is not None:
            preprocessor = FeatureStandardizer.fit_with_selection(x_train, y_train, k=config.feature_selection_k)
        else:
            preprocessor = FeatureStandardizer.fit(x_train)
            
        x_train_std = preprocessor.transform(x_train)
        x_val_std = preprocessor.transform(x_val)
        
        class_weight = balanced_class_weights(y_train)
        
        model, history = train_model(
            x_train_std,
            y_train,
            config,
            validation_data=(x_val_std, y_val),
            class_weight=class_weight,
        )
        
        if isinstance(model, NumpyBinaryClassifier):
            uncal_val_probs = predict_probability(model, x_val_std)
            a, b = fit_platt_scaling(uncal_val_probs, y_val)
            model.calibration_a = a
            model.calibration_b = b
            
        val_probs = predict_probability(model, x_val_std)
        threshold = optimize_threshold(y_val, val_probs)
        metrics = evaluate_predictions(y_val, val_probs, threshold)
        
        if "val_loss" in history and history["val_loss"]:
            metrics["validation_loss"] = float(history["val_loss"][-1])
        else:
            probs_clipped = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
            metrics["validation_loss"] = float(-np.mean(y_val * np.log(probs_clipped) + (1.0 - y_val) * np.log(1.0 - probs_clipped)))
            
        fold_metrics.append(metrics)
        
    cv_metrics = {}
    metric_keys = fold_metrics[0].keys()
    for key in metric_keys:
        values = [float(fm[key]) for fm in fold_metrics]
        cv_metrics[f"cv_mean_{key}"] = float(np.mean(values))
        cv_metrics[f"cv_std_{key}"] = float(np.std(values))
        
    result = train_single_model(features, labels, config)
    result.metrics.update(cv_metrics)
    result.metrics["cv_folds"] = n_splits
    
    return result
