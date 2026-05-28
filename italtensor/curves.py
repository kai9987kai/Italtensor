"""High-fidelity ASCII ROC and Precision-Recall curve generation and rendering."""

from __future__ import annotations

import math
import numpy as np

def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve coordinates."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    # Sort thresholds descending
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Find distinct thresholds
    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    # Cumulative sums
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add coordinates for threshold >= max(y_prob) + epsilon
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[y_prob_sorted[0] + 1e-5, y_prob_sorted[threshold_idxs]]

    tpr = tps / pos
    fpr = fps / neg

    # Cap to max 100 points for rendering performance
    if tpr.size > 100:
        indices = np.linspace(0, tpr.size - 1, 100, dtype=np.int32)
        tpr = tpr[indices]
        fpr = fpr[indices]
        thresholds = thresholds[indices]

    return fpr, tpr, thresholds


def compute_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall (PR) curve coordinates."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)

    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])

    desc_score_indices = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precision = tps / (tps + fps)
    recall = tps / pos

    # Insert end point: threshold at 0 has recall 1.0 and precision positive_prevalence
    prevalence = pos / y_true.size
    precision = np.r_[precision, prevalence]
    recall = np.r_[recall, 1.0]
    thresholds = np.r_[y_prob_sorted[threshold_idxs], 0.0]

    # Cap to max 100 points for rendering performance
    if precision.size > 100:
        indices = np.linspace(0, precision.size - 1, 100, dtype=np.int32)
        precision = precision[indices]
        recall = recall[indices]
        thresholds = thresholds[indices]

    return recall, precision, thresholds


def render_ascii_curve(
    title: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_label: str,
    y_label: str,
    width: int = 50,
    height: int = 15,
) -> str:
    """Plot custom x-y coordinates on a beautiful ASCII grid mapping."""
    grid = np.full((height, width), " ", dtype=object)

    # 1. Plot diagonal baseline reference if it is ROC curve (x_label starts with False)
    is_roc = "False" in x_label
    
    # 2. Add grid ticks at 0.25, 0.5, 0.75
    for r in range(height):
        y_val = r / (height - 1)
        # Check if row is near grid intervals
        if abs(y_val - 0.25) < 0.04 or abs(y_val - 0.5) < 0.04 or abs(y_val - 0.75) < 0.04:
            for c in range(width):
                grid[r, c] = "."

    # 3. Draw diagonal baseline for ROC
    if is_roc:
        for c in range(width):
            x_pct = c / (width - 1)
            r = int(round(x_pct * (height - 1)))
            if 0 <= r < height:
                grid[r, c] = "/"

    # 4. Map and plot the actual curve coordinates
    for x, y in zip(x_vals, y_vals, strict=True):
        c = int(round(x * (width - 1)))
        r = int(round(y * (height - 1)))
        c = np.clip(c, 0, width - 1)
        r = np.clip(r, 0, height - 1)
        grid[r, c] = "*"

    # 5. Build final text layout
    lines = [title]
    lines.append("-" * (width + 15))
    
    # Add border and y-axis ticks
    for r in range(height - 1, -1, -1):
        tick_val = r / (height - 1)
        row_str = "".join(grid[r, :])
        lines.append(f"{tick_val:>5.2f} | {row_str} |")
        
    lines.append("-" * (width + 15))
    
    # x-axis ticks
    ticks_str = " " * 8 + "0.00" + " " * (width // 4 - 3) + "0.25" + " " * (width // 4 - 3) + "0.50" + " " * (width // 4 - 3) + "0.75" + " " * (width // 4 - 4) + "1.00"
    lines.append(ticks_str)
    lines.append(f"Axis: Y = {y_label} vs X = {x_label}")
    lines.append("-" * (width + 15))

    return "\n".join(lines)


def render_evaluation_curves(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """Generate both ROC and PR curves in stacked ASCII layout."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)

    # Calculate AUCs
    from .experiments import _roc_auc, _average_precision
    auc_roc = _roc_auc(y_true, y_prob)
    auc_pr = _average_precision(y_true, y_prob)

    # ROC Curve
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    roc_ascii = render_ascii_curve(
        f"Receiver Operating Characteristic (ROC) Curve [AUC = {auc_roc:.4f}]",
        fpr,
        tpr,
        "False Positive Rate (FPR)",
        "True Positive Rate (TPR)",
    )

    # PR Curve
    recall, precision, _ = compute_pr_curve(y_true, y_prob)
    pr_ascii = render_ascii_curve(
        f"Precision-Recall (PR) Curve [AUC-PR / AP = {auc_pr:.4f}]",
        recall,
        precision,
        "Recall (TPR)",
        "Precision",
    )

    return f"{roc_ascii}\n\n{pr_ascii}"
