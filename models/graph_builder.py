"""Formal fixed spectral--spatial--feature KNN graph and local classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .regionizer import RegionOutput


@dataclass(frozen=True)
class GraphOutput:
    neighbor_indices: torch.Tensor
    neighbor_weights: torch.Tensor
    neighbor_mask: torch.Tensor | None = None
    token_coordinates: torch.Tensor | None = None


class WeightedResidualGraphLayer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.message_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(feature_dim)

    def forward(self, token_features: torch.Tensor, neighbor_indices: torch.Tensor,
                neighbor_weights: torch.Tensor, neighbor_mask: torch.Tensor | None = None) -> torch.Tensor:
        neighbor_features = token_features[:, neighbor_indices, :]
        weights = neighbor_weights[None] if neighbor_weights.ndim == 2 else neighbor_weights
        if neighbor_mask is not None:
            weights = weights * neighbor_mask.to(weights)[None]
        aggregated = (neighbor_features * weights[..., None]).sum(dim=2)
        return self.normalization(token_features + self.activation(self.message_projection(aggregated)))


class SpatialKNNGraphBuilder(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, grid_rows: int, grid_cols: int,
                 k_neighbors: int = 8, knn_chunk_size: int = 512, num_layers: int = 2,
                 graph_alpha: float = 1.0, graph_beta: float = 0.5,
                 graph_gamma: float = 0.5) -> None:
        super().__init__()
        token_count = int(grid_rows) * int(grid_cols)
        if not 0 < k_neighbors < token_count:
            raise ValueError("k_neighbors must be in [1, token_count - 1]")
        if num_layers != 2:
            raise ValueError("OCSFlow uses exactly two local graph layers")
        rows = torch.linspace(0.0, 1.0, int(grid_rows))
        cols = torch.linspace(0.0, 1.0, int(grid_cols))
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        centers = torch.stack((row_grid.reshape(-1), col_grid.reshape(-1)), dim=-1)
        self.graph_alpha = float(graph_alpha)
        self.graph_beta = float(graph_beta)
        self.graph_gamma = float(graph_gamma)
        self._configured_k_neighbors = int(k_neighbors)
        self._knn_chunk_size = int(knn_chunk_size)
        self.register_buffer("center_coordinates", centers, persistent=True)
        self.register_buffer("neighbor_indices", torch.empty((0, k_neighbors), dtype=torch.long), persistent=True)
        self.register_buffer("neighbor_weights", torch.empty((0, k_neighbors), dtype=centers.dtype), persistent=True)
        self.register_buffer("neighbor_spatial_distances", torch.empty((0, k_neighbors), dtype=centers.dtype), persistent=False)
        self.layers = nn.ModuleList([WeightedResidualGraphLayer(feature_dim) for _ in range(num_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim),
                                        nn.GELU(), nn.Linear(feature_dim, num_classes))

    @property
    def token_count(self) -> int:
        return int(self.center_coordinates.shape[0])

    @staticmethod
    def _chunked_exact_spatial_knn(centers: torch.Tensor, *, k_neighbors: int,
                                   chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = int(centers.shape[0])
        distances_out, indices_out = [], []
        candidates = torch.arange(token_count, device=centers.device)
        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            distances = torch.cdist(centers[start:end], centers, p=2,
                                    compute_mode="donot_use_mm_for_euclid_dist")
            distances[torch.arange(end - start, device=centers.device), candidates[start:end]] = float("inf")
            values, indices = torch.topk(distances, k=k_neighbors, dim=1, largest=False)
            distances_out.append(values)
            indices_out.append(indices)
        return torch.cat(distances_out), torch.cat(indices_out)

    def _ensure_spatial_topology(self) -> None:
        if self.neighbor_indices.shape[0] == self.token_count:
            return
        distances, indices = self._chunked_exact_spatial_knn(
            self.center_coordinates, k_neighbors=self._configured_k_neighbors,
            chunk_size=self._knn_chunk_size)
        inverse = 1.0 / distances.clamp_min(1e-8)
        self.neighbor_indices = indices
        self.neighbor_weights = inverse / inverse.sum(dim=1, keepdim=True)
        self.neighbor_spatial_distances = distances

    def forward(self, region_output: RegionOutput) -> GraphOutput:
        features = region_output.token_features
        spectra = region_output.token_spectra
        if spectra is None:
            raise ValueError("formal graph requires token spectra")
        if features.shape[1] != self.token_count:
            raise ValueError("token count does not match the configured grid")
        self._ensure_spatial_topology()
        with torch.no_grad():
            norm_spectra = F.normalize(spectra.detach(), dim=-1, eps=1e-8)
            norm_features = F.normalize(features.detach(), dim=-1, eps=1e-8)
            spectral_distance = 1.0 - (norm_spectra[:, :, None, :] * norm_spectra[:, self.neighbor_indices, :]).sum(-1).clamp(-1.0, 1.0)
            feature_distance = 1.0 - (norm_features[:, :, None, :] * norm_features[:, self.neighbor_indices, :]).sum(-1).clamp(-1.0, 1.0)
            distance = (self.graph_alpha * self.neighbor_spatial_distances[None]
                        + self.graph_beta * spectral_distance + self.graph_gamma * feature_distance)
            weights = torch.softmax(-distance, dim=-1)
            if weights.shape[0] == 1:
                weights = weights[0]
        return GraphOutput(self.neighbor_indices, weights,
                           token_coordinates=self.center_coordinates[None])

    def classify(self, region_output: RegionOutput, graph_output: GraphOutput) -> torch.Tensor:
        features = region_output.token_features
        for layer in self.layers:
            features = layer(features, graph_output.neighbor_indices,
                             graph_output.neighbor_weights, graph_output.neighbor_mask)
        return self.classifier(features)
