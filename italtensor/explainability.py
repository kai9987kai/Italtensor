"""Explainability and diagnostics: SHAP local attributions and decision boundary mapping."""

from __future__ import annotations

from typing import Any
import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer


def compute_shap_attributions(
    model: Any,
    sample: np.ndarray,
    preprocessor: FeatureStandardizer | None,
) -> list[dict[str, Any]]:
    """Compute local feature attributions (marginal contributions) using feature perturbation.
    
    Args:
        model: Trained classifier.
        sample: 1D array of shape (D,) representing the feature vector of a single sample.
        preprocessor: Optional feature standardizer.
        
    Returns:
        A list of dicts containing attribution metadata for each feature.
    """
    x = np.asarray(sample, dtype=np.float32).reshape(-1)
    dim = x.shape[0]
    
    # 1. Base prediction probability
    x_input = x.reshape(1, -1)
    x_prep = preprocessor.transform(x_input) if preprocessor is not None else x_input
    base_p = float(predict_probability(model, x_prep)[0])
    
    # 2. Perturbation loop
    # For each feature, we set its value to the "neutral" mean/median (preprocessor.mean)
    # or zero if no preprocessor is provided, and measure change in prediction.
    attributions = []
    
    # Extract feature names if present
    feature_names = [f"x{i+1}" for i in range(dim)]
    selected_indices = preprocessor.selected_indices if preprocessor is not None else None
    
    for j in range(dim):
        x_pert = x.copy()
        if preprocessor is not None and preprocessor.mean is not None:
            # Check if this feature is active or was filtered out
            if selected_indices is not None:
                # If feature is in selected indices, replace with its mean
                if j in selected_indices:
                    mapped_j = selected_indices.index(j)
                    x_pert[j] = float(preprocessor.mean[mapped_j])
                else:
                    # Filtered feature - changing it does nothing as model doesn't see it
                    x_pert[j] = 0.0
            else:
                x_pert[j] = float(preprocessor.mean[j])
        else:
            x_pert[j] = 0.0
            
        x_pert_input = x_pert.reshape(1, -1)
        x_pert_prep = preprocessor.transform(x_pert_input) if preprocessor is not None else x_pert_input
        pert_p = float(predict_probability(model, x_pert_prep)[0])
        
        # Attribution: original prediction minus prediction with feature neutralized
        # A positive value means the feature's actual value pulled the model towards class 1
        attribution_val = base_p - pert_p
        
        attributions.append(
            {
                "feature_index": int(j),
                "feature_name": feature_names[j],
                "actual_value": float(x[j]),
                "neutral_value": float(x_pert[j]),
                "attribution": attribution_val,
            }
        )
        
    return attributions


def render_shap_bar_chart(attributions: list[dict[str, Any]], max_width: int = 24) -> str:
    """Render a text-based ASCII bar chart representing local SHAP attributions."""
    if not attributions:
        return "No attributions to display."
        
    lines = ["SHAP Local Feature Attributions (Contribution to class 1 probability):"]
    lines.append("-" * 75)
    
    max_att = max(abs(item["attribution"]) for item in attributions)
    if max_att < 1e-6:
        max_att = 1.0  # Avoid division by zero
        
    for item in attributions:
        att = item["attribution"]
        val = item["actual_value"]
        name = item["feature_name"]
        
        # Calculate visual bar width
        bar_len = int(round((abs(att) / max_att) * max_width))
        
        if att >= 0:
            bar = "=" * bar_len + ">"
            spaces = " " * (max_width - bar_len)
            visual = f"{spaces}{bar} (+{att:.4f})"
        else:
            bar = "<" + "=" * bar_len
            spaces = " " * (max_width - bar_len)
            visual = f"({att:.4f}) {bar}{spaces}"
            
        lines.append(f"{name:<10} [val={val:>7.3f}]: {visual}")
        
    lines.append("-" * 75)
    return "\n".join(lines)


def generate_decision_boundary_map(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    preprocessor: FeatureStandardizer | None,
    resolution: int = 40,
) -> str:
    """Project dataset to 2D using PCA and draw a beautiful ASCII grid boundary map."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    
    if x.ndim != 2 or x.shape[0] == 0:
        return "Decision boundary visualizer requires a non-empty 2D feature matrix."
        
    n_samples, dim = x.shape
    
    # 1. PCA Projection to 2D
    mu = np.mean(x, axis=0)
    x_centered = x - mu
    
    if dim == 1:
        # Trivial 1D case, duplicate feature to make it 2D
        w = np.array([[1.0, 0.0]], dtype=np.float32)
        x_2d = np.c_[x_centered, np.zeros_like(x_centered)]
    else:
        cov = np.dot(x_centered.T, x_centered) / max(1, n_samples - 1)
        # Use SVD or eigh for eigenvalues
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort descending
            sort_idx = np.argsort(eigenvalues)[::-1]
            w = eigenvectors[:, sort_idx[:2]]
        except np.linalg.LinAlgError:
            # Fallback to random projection if covariance is singular or eigh fails
            w = np.zeros((dim, 2), dtype=np.float32)
            w[0, 0] = 1.0
            w[min(1, dim - 1), 1] = 1.0
            
        x_2d = np.dot(x_centered, w)
        
    # 2. Grid bounds in 2D projected space
    x_min, x_max = x_2d[:, 0].min() - 0.5, x_2d[:, 0].max() + 0.5
    y_min, y_max = x_2d[:, 1].min() - 0.5, x_2d[:, 1].max() + 0.5
    
    # Ensure some spread
    if abs(x_max - x_min) < 1e-4:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if abs(y_max - y_min) < 1e-4:
        y_min, y_max = y_min - 1.0, y_max + 1.0
        
    grid_x = np.linspace(x_min, x_max, resolution)
    grid_y = np.linspace(y_min, y_max, resolution)
    
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # 3. Project grid points back to N-D
    if dim == 1:
        grid_nd = grid_2d[:, :1] + mu
    else:
        grid_nd = np.dot(grid_2d, w.T) + mu
        
    # 4. Predict probabilities on the grid
    grid_prep = preprocessor.transform(grid_nd) if preprocessor is not None else grid_nd
    probs = predict_probability(model, grid_prep).reshape(resolution, resolution)
    
    # 5. Populate character matrix
    grid_chars = np.empty((resolution, resolution), dtype=object)
    for r in range(resolution):
        for c in range(resolution):
            p = probs[r, c]
            if abs(p - 0.5) < 0.06:
                grid_chars[r, c] = "x"  # Decision boundary zone
            elif p > 0.5:
                grid_chars[r, c] = "*"  # Class 1 prediction
            else:
                grid_chars[r, c] = "."  # Class 0 prediction
                
    # 6. Scatter actual samples onto the grid
    for i in range(n_samples):
        px, py = x_2d[i, 0], x_2d[i, 1]
        c_idx = int(np.argmin(np.abs(grid_x - px)))
        r_idx = int(np.argmin(np.abs(grid_y - py)))
        
        # Overlay actual labels
        label_char = "0" if y[i] == 0 else "1"
        grid_chars[r_idx, c_idx] = label_char
        
    # 7. Render map output
    lines = ["Decision Boundary Visualization (PCA Projected 2D Space):"]
    lines.append(f"y-axis: PC2 ({y_min:.2f} to {y_max:.2f})  |  x-axis: PC1 ({x_min:.2f} to {x_max:.2f})")
    lines.append("-" * (resolution + 25))
    lines.append("Legend:  . = Predict 0 | * = Predict 1 | x = Decision Boundary | 0/1 = Actual Samples")
    lines.append("-" * (resolution + 25))
    
    for r in range(resolution - 1, -1, -1):
        row_str = "".join(grid_chars[r, :])
        lines.append(f"PC2={grid_y[r]:>6.2f} | {row_str} |")
        
    lines.append("-" * (resolution + 25))
    return "\n".join(lines)
