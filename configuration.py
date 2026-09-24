"""Load the single release YAML and resolve one dataset's runtime settings."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from models.model import OCSFlow


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, dataset: str, seed: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("common"), dict):
        raise ValueError("config must contain a common mapping")
    code = dataset.strip().upper()
    datasets = document.get("datasets")
    if not isinstance(datasets, dict) or code not in datasets:
        raise ValueError(f"config has no dataset entry for {code}")
    resolved = _deep_merge(document["common"], datasets[code])
    resolved["dataset"] = code
    resolved["seed"] = int(seed)
    return resolved


def resolve_runtime_config(
    config: dict[str, Any],
    *,
    image_shape: tuple[int, int, int],
    num_classes: int,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    height, width, bands = image_shape
    model = resolved["model"]
    if "grid_rows" not in model or "grid_cols" not in model:
        auto = resolved["auto_grid"]
        row_stride = float(auto["row_stride"])
        col_stride = float(auto["col_stride"])
        if row_stride <= 0 or col_stride <= 0:
            raise ValueError("auto-grid strides must be positive")
        model["grid_rows"] = max(int(auto.get("min_rows", 3)), int(round(height / row_stride)))
        model["grid_cols"] = max(int(auto.get("min_cols", 3)), int(round(width / col_stride)))
    model["in_channels"] = int(bands)
    model["num_classes"] = int(num_classes)
    return resolved


def build_model(config: dict[str, Any]) -> OCSFlow:
    model = config["model"]
    return OCSFlow(
        in_channels=int(model["in_channels"]),
        num_classes=int(model["num_classes"]),
        feature_dim=int(model["feature_dim"]),
        group_norm_groups=int(model["group_norm_groups"]),
        grid_rows=int(model["grid_rows"]),
        grid_cols=int(model["grid_cols"]),
        temperature=float(model["temperature"]),
        spatial_weight=float(model["spatial_weight"]),
        spectral_weight=float(model["spectral_weight"]),
        feature_weight=float(model["feature_weight"]),
        center_pool_size=int(model["center_pool_size"]),
        chunk_size=int(model["chunk_size"]),
        k_neighbors=int(model["k_neighbors"]),
        knn_chunk_size=int(model["knn_chunk_size"]),
        graph_layers=int(model["graph_layers"]),
        fm_hidden_dim=int(model["fm_hidden_dim"]),
        sigma=float(model["sigma"]),
        euler_steps=int(model["euler_steps"]),
        graph_alpha=float(model["graph_alpha"]),
        graph_beta=float(model["graph_beta"]),
        graph_gamma=float(model["graph_gamma"]),
        residual_scale_init=float(model["residual_scale_init"]),
        transformer_attention=str(model["transformer_attention"]),
        boundary=config["boundary_candidate_reweight"],
        remote=config["sparse_remote_token_coupling"],
    )
