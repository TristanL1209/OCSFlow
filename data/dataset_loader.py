"""MAT/HDF5 loading for IP, PU, KSC and HC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

from .registry import DatasetSpec, get_dataset_spec


@dataclass(frozen=True)
class SingleImageDataset:
    image: np.ndarray
    ground_truth: np.ndarray
    dataset_name: str
    num_classes: int


def _is_hdf5(path: Path) -> bool:
    return h5py.is_hdf5(path)


def _variables(path: Path, ndim: int) -> list[tuple[str, tuple[int, ...], int]]:
    if not path.is_file():
        raise FileNotFoundError(f"required dataset file not found: {path}")
    if _is_hdf5(path):
        found: list[tuple[str, tuple[int, ...], int]] = []
        with h5py.File(path, "r") as handle:
            def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                    shape = tuple(reversed(tuple(int(v) for v in obj.shape)))
                    if len(shape) == ndim:
                        found.append((name, shape, int(np.prod(shape))))
            handle.visititems(visit)
        return found
    return [(name, tuple(int(v) for v in shape), int(np.prod(shape)))
            for name, shape, kind in whosmat(path)
            if len(shape) == ndim and kind.lower() not in {"struct", "cell", "char", "object"}]


def _primary(path: Path, ndim: int) -> tuple[str, tuple[int, ...]]:
    candidates = sorted(_variables(path, ndim), key=lambda item: item[2], reverse=True)
    if not candidates:
        raise KeyError(f"{path.name}: no numeric {ndim}-D array")
    if len(candidates) > 1 and candidates[0][2] == candidates[1][2]:
        raise RuntimeError(f"{path.name}: ambiguous largest numeric array")
    return candidates[0][0], candidates[0][1]


def _load(path: Path, key: str) -> np.ndarray:
    if _is_hdf5(path):
        with h5py.File(path, "r") as handle:
            array = np.asarray(handle[key])
        return array.transpose(tuple(reversed(range(array.ndim))))
    return np.asarray(loadmat(path, variable_names=[key])[key])


def _axis_order(image_shape: tuple[int, int, int], gt_shape: tuple[int, int]) -> tuple[int, int, int]:
    matches = []
    for band_axis in range(3):
        spatial = [axis for axis in range(3) if axis != band_axis]
        for first, second in (tuple(spatial), tuple(reversed(spatial))):
            order = (first, second, band_axis)
            if tuple(image_shape[index] for index in order)[:2] == gt_shape:
                matches.append(order)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    if unique and gt_shape[0] == gt_shape[1] and len({item[2] for item in unique}) == 1:
        return min(unique)
    raise ValueError(f"cannot uniquely orient image {image_shape} to GT {gt_shape}")


def load_dataset(name: str, data_root: str | Path) -> SingleImageDataset:
    code = name.strip().upper()
    spec: DatasetSpec = get_dataset_spec(code)
    root = Path(data_root)
    gt_path, image_path = root / spec.gt_file, root / spec.image_file
    gt_key, _ = _primary(gt_path, 2)
    ground_truth = np.asarray(_load(gt_path, gt_key)).squeeze()
    if not np.issubdtype(ground_truth.dtype, np.integer):
        rounded = np.rint(ground_truth)
        if not np.array_equal(ground_truth, rounded):
            raise ValueError("ground truth contains non-integer labels")
        ground_truth = rounded
    ground_truth = ground_truth.astype(np.int64, copy=False)
    if int(ground_truth.max(initial=0)) != spec.num_classes:
        raise ValueError(f"{code}: expected labels 0..{spec.num_classes}")
    image_key, image_shape = _primary(image_path, 3)
    image = np.transpose(_load(image_path, image_key),
                         _axis_order(image_shape, tuple(ground_truth.shape)))
    return SingleImageDataset(np.asarray(image), ground_truth, code, spec.num_classes)
