"""Reproducible stratified pixel splits for sparse HSI supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from math import ceil


@dataclass(frozen=True)
class SplitMasks:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def validate_split_masks(ground_truth: np.ndarray, masks: SplitMasks) -> None:
    gt = np.asarray(ground_truth)
    if gt.ndim != 2:
        raise ValueError(f"ground_truth must be 2-D, got {gt.shape}")

    named = {
        "train_mask": np.asarray(masks.train_mask),
        "val_mask": np.asarray(masks.val_mask),
        "test_mask": np.asarray(masks.test_mask),
    }
    for name, mask in named.items():
        if mask.shape != gt.shape:
            raise ValueError(f"{name} shape {mask.shape} does not match ground truth {gt.shape}")
        if mask.dtype != np.bool_:
            raise TypeError(f"{name} must have boolean dtype, got {mask.dtype}")
        if np.any(mask & (gt == 0)):
            raise ValueError(f"{name} contains background pixels (label 0)")

    train, val, test = named.values()
    if np.any(train & val) or np.any(train & test) or np.any(val & test):
        raise ValueError("train_mask, val_mask and test_mask must be mutually disjoint")
    covered = train | val | test
    if not np.array_equal(covered, gt > 0):
        raise ValueError("Every labeled pixel must belong to exactly one split")


def stratified_random_split(
    ground_truth: np.ndarray,
    train_per_class: int = 30,
    val_per_class: int = 10,
    seed: int = 0,
) -> SplitMasks:
    """Split every positive class exactly; label 0 is always excluded.

    For classes with at least ``train_per_class + val_per_class`` pixels, the
    requested protocol is used exactly.

    For small classes with fewer than ``train_per_class`` pixels, the class is
    split into roughly half training pixels and the remainder is split across
    validation and test. This keeps rare classes usable while preserving the
    train/validation/test separation.

    Classes in the intermediate range ``[train_per_class, train_per_class +
    val_per_class)`` still raise a clear error because they satisfy the
    training request but not the validation request.
    """
    gt = np.asarray(ground_truth)
    if gt.ndim != 2:
        raise ValueError(f"ground_truth must have shape [H, W], got {gt.shape}")
    if train_per_class < 0 or val_per_class < 0:
        raise ValueError("train_per_class and val_per_class must be non-negative")
    if np.any(gt < 0):
        raise ValueError("Raw ground truth may not contain negative labels")

    classes = np.unique(gt[gt > 0])
    if classes.size == 0:
        raise ValueError("ground_truth contains no labeled classes (labels > 0)")

    required = train_per_class + val_per_class
    rng = np.random.default_rng(seed)
    flat_gt = gt.reshape(-1)
    train = np.zeros(flat_gt.shape, dtype=bool)
    val = np.zeros(flat_gt.shape, dtype=bool)
    test = np.zeros(flat_gt.shape, dtype=bool)

    for class_id in classes:
        indices = np.flatnonzero(flat_gt == class_id)
        if indices.size < train_per_class:
            train_count = max(1, int(ceil(indices.size / 2.0)))
            remaining = int(indices.size - train_count)
            val_count = remaining // 2
            test_count = remaining - val_count
        elif indices.size < required:
            raise ValueError(
                f"Class {int(class_id)} has {indices.size} labeled pixels, "
                f"but {required} are required ({train_per_class} train + "
                f"{val_per_class} validation)."
            )
        else:
            train_count = int(train_per_class)
            val_count = int(val_per_class)
            test_count = int(indices.size - train_count - val_count)
        shuffled = rng.permutation(indices)
        train[shuffled[:train_count]] = True
        val_start = train_count
        val_stop = val_start + val_count
        test_stop = val_stop + test_count
        val[shuffled[val_start:val_stop]] = True
        test[shuffled[val_stop:test_stop]] = True

    masks = SplitMasks(
        train_mask=train.reshape(gt.shape),
        val_mask=val.reshape(gt.shape),
        test_mask=test.reshape(gt.shape),
    )
    validate_split_masks(gt, masks)
    return masks
