import numpy as np
import pytest
from evaluation.metrics import compute_multiclass_metrics, compute_expected_calibration_error


def test_compute_multiclass_metrics():
    y_true = ["sparrow", "dove", "sparrow", "cicada"]
    y_pred = ["sparrow", "dove", "sparrow", "sparrow"]

    metrics = compute_multiclass_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert metrics["total_samples"] == 4
    assert len(metrics["classes"]) == 3


def test_expected_calibration_error():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    y_prob = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    ece = compute_expected_calibration_error(y_true, y_pred, y_prob, ["A", "B"])
    assert ece == pytest.approx(0.0, abs=1e-5)
