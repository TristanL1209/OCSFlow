"""Formal train-pixel z-score normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def normalize_train_pixels(image: np.ndarray, train_mask: np.ndarray,
                           *, eps: float = 1e-8) -> tuple[np.ndarray, NormalizationStats]:
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool)
    if image.ndim != 3 or mask.shape != image.shape[:2]:
        raise ValueError("image must be [H,W,B] and train_mask [H,W]")
    samples = image[mask]
    if samples.size == 0:
        raise ValueError("train_mask selects no pixels")
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = samples.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, float(eps)).astype(np.float32)
    normalized = (image - mean[None, None, :]) / std[None, None, :]
    return normalized.astype(np.float32, copy=False), NormalizationStats(mean, std)
