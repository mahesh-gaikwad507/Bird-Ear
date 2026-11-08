from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate classification metrics including macro and weighted scores.
    """
    labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    result = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classes": labels,
        "confusion_matrix": cm,
        "per_class_report": report,
        "total_samples": len(y_true),
    }

    # Confidence calibration error if probabilities provided
    if y_prob is not None:
        ece = compute_expected_calibration_error(y_true, y_pred, y_prob, labels)
        result["expected_calibration_error"] = round(ece, 4)

    return result


def compute_expected_calibration_error(
    y_true: List[str],
    y_pred: List[str],
    y_prob: np.ndarray,
    classes: List[str],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    max_probs = np.max(y_prob, axis=1)
    correct = np.array([yt == yp for yt, yp in zip(y_true, y_pred)], dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        in_bin = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(max_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
