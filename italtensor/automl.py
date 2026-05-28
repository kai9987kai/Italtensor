"""Tree-structured Parzen Estimator (TPE) AutoML optimization in pure NumPy."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence
import numpy as np

from .modeling import ModelConfig
from .experiments import train_single_model, train_single_model_cv, ExperimentResult

class HyperparameterSpace:
    """Defines the search space domains for hyperparameters."""
    def __init__(self, n_features: int):
        self.n_features = n_features
        # Define domains: (type, options/bounds)
        # 'continuous_log': (min_val, max_val)
        # 'continuous_uniform': (min_val, max_val)
        # 'discrete_choice': list of options
        self.space: dict[str, tuple[str, Any]] = {
            "learning_rate": ("continuous_log", (1e-4, 1e-1)),
            "l1_penalty": ("continuous_uniform", (0.0, 0.1)),
            "gradient_clip": ("continuous_uniform", (0.0, 5.0)),
            "max_epochs": ("discrete_choice", [10, 20, 50, 100]),
            "batch_size": ("discrete_choice", [8, 16, 32, 64]),
            "feature_selection_k": ("discrete_choice", self._get_feature_k_choices()),
            "smote_k": ("discrete_choice", [1, 3, 5, 7]),
        }

    def _get_feature_k_choices(self) -> list[int | None]:
        choices: list[int | None] = [None]
        for k in [2, 5, 10, 15, 20]:
            if k < self.n_features:
                choices.append(k)
        return choices

    def sample_random(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw a random sample from the search space."""
        params = {}
        for name, (space_type, bounds) in self.space.items():
            if space_type == "continuous_log":
                log_min, log_max = math.log(bounds[0]), math.log(bounds[1])
                params[name] = float(math.exp(rng.uniform(log_min, log_max)))
            elif space_type == "continuous_uniform":
                params[name] = float(rng.uniform(bounds[0], bounds[1]))
            elif space_type == "discrete_choice":
                idx = rng.choice(len(bounds))
                params[name] = bounds[idx]
        return params


def gaussian_kde_density(x: float, samples: np.ndarray, bandwidth: float) -> float:
    """Compute Gaussian Kernel Density Estimation at a single point x."""
    if samples.size == 0:
        return 1.0
    diff = samples - x
    # Standard normal PDF
    pdf_vals = np.exp(-0.5 * (diff / bandwidth) ** 2) / (math.sqrt(2 * math.pi) * bandwidth)
    return float(np.mean(pdf_vals))


def estimate_bandwidth(samples: np.ndarray, min_bw: float = 1e-3) -> float:
    """Scott's rule of thumb to estimate KDE bandwidth."""
    n = samples.size
    if n <= 1:
        return min_bw
    std = float(np.std(samples, ddof=1))
    if std < min_bw:
        std = min_bw
    # Scott's Rule: std * n**(-1/5)
    return max(std * (n ** (-0.2)), min_bw)


class TPEOptimizer:
    """Tree-structured Parzen Estimator Optimizer in pure NumPy."""
    def __init__(self, space: HyperparameterSpace, gamma: float = 0.25, n_startup: int = 5, seed: int = 42):
        self.space = space
        self.gamma = gamma
        self.n_startup = n_startup
        self.rng = np.random.default_rng(seed)
        self.trials: list[dict[str, Any]] = []

    def register_trial(self, params: dict[str, Any], score: float) -> None:
        """Register the result of a hyperparameter evaluation trial."""
        self.trials.append({"params": params, "score": score})

    def suggest(self) -> dict[str, Any]:
        """Suggest a new set of hyperparameters using TPE."""
        n_trials = len(self.trials)
        if n_trials < self.n_startup:
            return self.space.sample_random(self.rng)

        # 1. Sort trials by score (descending, since higher is better)
        sorted_trials = sorted(self.trials, key=lambda t: t["score"], reverse=True)
        
        # 2. Split trials into good and bad sets
        n_good = max(1, int(math.ceil(self.gamma * n_trials)))
        good_trials = sorted_trials[:n_good]
        bad_trials = sorted_trials[n_good:]

        suggested_params = {}

        # 3. For each hyperparameter, build the estimator and choose the best candidate
        for name, (space_type, bounds) in self.space.space.items():
            good_vals = np.array([t["params"][name] for t in good_trials], dtype=np.float32)
            bad_vals = np.array([t["params"][name] for t in bad_trials], dtype=np.float32)

            if space_type in ("continuous_log", "continuous_uniform"):
                # Work in log-space for log-uniform
                is_log = space_type == "continuous_log"
                if is_log:
                    good_vals = np.log(np.clip(good_vals, 1e-7, None))
                    bad_vals = np.log(np.clip(bad_vals, 1e-7, None))
                    min_bound, max_bound = math.log(bounds[0]), math.log(bounds[1])
                else:
                    min_bound, max_bound = bounds[0], bounds[1]

                # Estimate bandwidths
                good_bw = estimate_bandwidth(good_vals, min_bw=(max_bound - min_bound) * 0.05)
                bad_bw = estimate_bandwidth(bad_vals, min_bw=(max_bound - min_bound) * 0.05)

                # Generate candidates from the good KDE
                candidates: list[float] = []
                for _ in range(24):
                    # Pick a random sample from good_vals and perturb
                    source_val = float(self.rng.choice(good_vals))
                    noise = self.rng.normal(0.0, good_bw)
                    cand = float(np.clip(source_val + noise, min_bound, max_bound))
                    candidates.append(cand)

                # Evaluate densities of candidates under l(x) and g(x)
                best_cand = candidates[0]
                best_ratio = -1.0
                for cand in candidates:
                    l_x = gaussian_kde_density(cand, good_vals, good_bw)
                    g_x = gaussian_kde_density(cand, bad_vals, bad_bw)
                    ratio = l_x / max(g_x, 1e-12)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_cand = cand

                if is_log:
                    suggested_params[name] = float(math.exp(best_cand))
                else:
                    suggested_params[name] = float(best_cand)

            elif space_type == "discrete_choice":
                # For discrete choices, compute frequencies with Laplace smoothing
                # options is the list of bounds
                options = bounds
                
                # Helper to convert None to a string key or similar for counts
                def key_map(v: Any) -> str:
                    return "None" if v is None else str(v)

                good_counts = {key_map(opt): 0.5 for opt in options}
                bad_counts = {key_map(opt): 0.5 for opt in options}

                for v in good_vals:
                    good_counts[key_map(v)] += 1.0
                for v in bad_vals:
                    bad_counts[key_map(v)] += 1.0

                total_good = sum(good_counts.values())
                total_bad = sum(bad_counts.values())

                good_probs = {k: v / total_good for k, v in good_counts.items()}
                bad_probs = {k: v / total_bad for k, v in bad_counts.items()}

                # We can sample a candidate from the good distribution
                # Pick 24 candidates and select the one maximizing l(x) / g(x)
                choice_keys = [key_map(opt) for opt in options]
                p_good = [good_probs[k] for k in choice_keys]
                
                candidates_keys = self.rng.choice(choice_keys, size=24, p=p_good)
                
                best_key = candidates_keys[0]
                best_ratio = -1.0
                for key in candidates_keys:
                    l_x = good_probs[key]
                    g_x = bad_probs[key]
                    ratio = l_x / max(g_x, 1e-12)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_key = key

                # Convert back from key
                idx = choice_keys.index(best_key)
                suggested_params[name] = options[idx]

        return suggested_params


def run_auto_tune(
    features: np.ndarray,
    labels: np.ndarray,
    n_trials: int,
    base_config: ModelConfig,
    use_cv: bool = True,
    tune_lr: bool = True,
    tune_l1: bool = True,
    tune_smote: bool = True,
    tune_features: bool = True,
    logger: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], ExperimentResult]:
    """Execute hyperparameter optimization sweep using pure NumPy TPE.
    
    Args:
        features: 2D feature matrix
        labels: 1D label array
        n_trials: Number of optimization trials to run
        base_config: Starting baseline ModelConfig
        use_cv: If True, uses cross-validation; otherwise splits once
        tune_lr: Tune learning_rate
        tune_l1: Tune l1_penalty
        tune_smote: Tune use_smote and smote_k
        tune_features: Tune feature_selection_k
        logger: Logging callback for printing progress
        
    Returns:
        A tuple of (best_parameters, best_experiment_result)
    """
    def log(msg: str):
        if logger is not None:
            logger(msg)

    n_samples, n_features = features.shape
    space = HyperparameterSpace(n_features)
    optimizer = TPEOptimizer(space, seed=base_config.random_seed)

    best_score = -1.0
    best_params = {}
    best_result: ExperimentResult | None = None

    log(f"Starting AutoML Hyperparameter Tuning Sweep ({n_trials} trials)...")
    log(f"Optimization metric: Validation F1-Score (higher is better)")

    for trial_idx in range(1, n_trials + 1):
        # 1. Suggest parameters
        suggested = optimizer.suggest()
        
        # 2. Merge with baseline configuration based on what is active for tuning
        trial_params = base_config.to_dict()
        active_tune_keys = []

        if tune_lr:
            trial_params["learning_rate"] = suggested["learning_rate"]
            active_tune_keys.append(f"learning_rate={suggested['learning_rate']:.5f}")
        if tune_l1:
            trial_params["l1_penalty"] = suggested["l1_penalty"]
            active_tune_keys.append(f"l1_penalty={suggested['l1_penalty']:.5f}")
            trial_params["gradient_clip"] = suggested["gradient_clip"]
            active_tune_keys.append(f"gradient_clip={suggested['gradient_clip']:.3f}")
        
        # Epochs/batch_size are optimized as part of base tuning
        trial_params["max_epochs"] = suggested["max_epochs"]
        trial_params["batch_size"] = suggested["batch_size"]
        active_tune_keys.append(f"epochs={suggested['max_epochs']}")
        active_tune_keys.append(f"batch_size={suggested['batch_size']}")

        if tune_features:
            trial_params["feature_selection_k"] = suggested["feature_selection_k"]
            active_tune_keys.append(f"feat_k={suggested['feature_selection_k']}")

        use_smote = False
        smote_k = 3
        if tune_smote:
            use_smote = True
            smote_k = int(suggested["smote_k"])
            active_tune_keys.append(f"SMOTE_k={smote_k}")

        trial_config = ModelConfig.from_dict(trial_params)

        params_str = ", ".join(active_tune_keys)
        log(f"\n[Trial {trial_idx}/{n_trials}] Testing: {params_str}")

        try:
            # 3. Evaluate configuration
            if use_cv:
                # Use 5-fold CV to get a stable score
                result = train_single_model_cv(
                    features,
                    labels,
                    trial_config,
                    n_splits=5,
                    use_smote=use_smote,
                    smote_k=smote_k,
                )
            else:
                result = train_single_model(
                    features,
                    labels,
                    trial_config,
                    use_smote=use_smote,
                    smote_k=smote_k,
                )

            # Optimization metric is F1 score
            score = float(result.metrics.get("f1", 0.0))
            loss = float(result.metrics.get("validation_loss", 0.0))
            
            log(f"Trial {trial_idx} Result: F1-Score={score:.4f}, Val Loss={loss:.4f}")

            # Register score back to Parzen Estimator optimizer
            optimizer.register_trial(suggested, score)

            # Track overall best
            if score > best_score:
                best_score = score
                best_params = {
                    "learning_rate": trial_config.learning_rate,
                    "l1_penalty": trial_config.l1_penalty,
                    "gradient_clip": trial_config.gradient_clip,
                    "max_epochs": trial_config.max_epochs,
                    "batch_size": trial_config.batch_size,
                    "feature_selection_k": trial_config.feature_selection_k,
                    "use_smote": use_smote,
                    "smote_k": smote_k,
                }
                best_result = result

        except Exception as exc:
            log(f"Trial {trial_idx} Failed: {exc}")
            # Register a low score for failures
            optimizer.register_trial(suggested, 0.0)

    if best_result is None:
        raise ValueError("AutoML Sweep completed but no successful trials were evaluated.")

    log("\n" + "=" * 60)
    log("AutoML Tuning Sweep Completed Successfully!")
    log(f"Best Validation F1-Score: {best_score:.4f}")
    log("Optimized Parameters:")
    for k, v in best_params.items():
        log(f"  - {k}: {v}")
    log("=" * 60 + "\n")

    return best_params, best_result
