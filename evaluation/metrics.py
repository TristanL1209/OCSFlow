"""OA, AA, kappa, macro-F1 and confusion matrix for labels 1..K."""

from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(ground_truth: np.ndarray, prediction: np.ndarray,
                           num_classes: int) -> dict[str, Any]:
    truth = np.asarray(ground_truth, dtype=np.int64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("evaluation arrays must be non-empty with identical shapes")
    encoded = (truth - 1) * num_classes + (pred - 1)
    matrix = np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    true_count, predicted_count = matrix.sum(1), matrix.sum(0)
    true_positive = np.diag(matrix).astype(np.float64)
    total = int(matrix.sum())
    per_class = np.divide(true_positive, true_count, out=np.full(num_classes, np.nan), where=true_count > 0)
    oa = float(true_positive.sum() / total)
    expected = float(np.dot(true_count, predicted_count) / (total * total))
    denominator = 2.0 * true_positive + predicted_count - true_positive + true_count - true_positive
    f1 = np.divide(2.0 * true_positive, denominator, out=np.full(num_classes, np.nan), where=denominator > 0)
    return {"oa": oa, "aa": float(np.nanmean(per_class)),
            "kappa": float((oa - expected) / (1.0 - expected)) if expected < 1.0 else 1.0,
            "macro_f1": float(np.nanmean(f1)),
            "per_class_accuracy": [None if np.isnan(v) else float(v) for v in per_class],
            "confusion_matrix": matrix.tolist(), "num_samples": total}
