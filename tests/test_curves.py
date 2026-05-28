import numpy as np
import pytest

from italtensor.curves import compute_roc_curve, compute_pr_curve, render_ascii_curve, render_evaluation_curves

def test_compute_curves():
    # Simple binary test set
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    
    fpr, tpr, thresholds_roc = compute_roc_curve(y_true, y_prob)
    assert fpr.size > 0
    assert tpr.size > 0
    assert fpr[0] == 0.0
    assert tpr[-1] == 1.0
    
    recall, precision, thresholds_pr = compute_pr_curve(y_true, y_prob)
    assert recall.size > 0
    assert precision.size > 0
    assert recall[0] >= 0.0
    assert recall[-1] == 1.0


def test_render_ascii_curve():
    x = np.linspace(0.0, 1.0, 10)
    y = x ** 2
    
    ascii_map = render_ascii_curve(
        "Test Curve",
        x,
        y,
        "X axis",
        "Y axis",
        width=30,
        height=10,
    )
    
    assert "Test Curve" in ascii_map
    assert "0.00" in ascii_map
    assert "1.00" in ascii_map
    assert "Axis: Y = Y axis vs X = X axis" in ascii_map


def test_render_evaluation_curves():
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    
    output = render_evaluation_curves(y_true, y_prob)
    assert "Receiver Operating Characteristic" in output
    assert "Precision-Recall" in output
