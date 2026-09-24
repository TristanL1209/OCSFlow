
"""Sparse local soft assignment from pixels to regular-grid supertokens."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RegionOutput:
    assignment_weights: torch.Tensor
    assignment_indices: torch.Tensor
    token_features: torch.Tensor
    token_spectra: torch.Tensor | None
    token_mass: torch.Tensor
    empty_token_count: int
    mean_assignment_entropy: torch.Tensor
    token_coordinates: torch.Tensor | None = None
    roi_pixel_indices: torch.Tensor | None = None
    full_pixel_count: int | None = None


class TokenRegionizer(nn.Module):
    """Base interface for token regionizers."""


class GridSoftRegionizer(TokenRegionizer):
    """Associate every pixel with only its nine nearest grid centers."""

    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        temperature: float = 0.5,
        spatial_weight: float = 10.0,
        spectral_weight: float = 1.0,
        feature_weight: float = 1.0,
        center_pool_size: int = 1,
        chunk_size: int = 8192,
        empty_mass_threshold: float = 1e-6,
        compute_token_spectra: bool = True,
    ) -> None:
        super().__init__()
        if grid_rows < 3 or grid_cols < 3:
            raise ValueError("grid_rows and grid_cols must both be at least 3")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if center_pool_size not in {1, 3, 5}:
            raise ValueError("center_pool_size must be one of {1, 3, 5}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.temperature = float(temperature)
        self.spatial_weight = float(spatial_weight)
        self.spectral_weight = float(spectral_weight)
        self.feature_weight = float(feature_weight)
        self.center_pool_size = int(center_pool_size)
        self.chunk_size = int(chunk_size)
        self.empty_mass_threshold = float(empty_mass_threshold)
        self.compute_token_spectra = bool(compute_token_spectra)
        self._spatial_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def num_tokens(self) -> int:
        return self.grid_rows * self.grid_cols

    def _spatial_structure(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (height, width, str(device))
        cached = self._spatial_cache.get(key)
        if cached is not None:
            return cached

        center_rows = torch.linspace(0, height - 1, self.grid_rows, device=device)
        center_cols = torch.linspace(0, width - 1, self.grid_cols, device=device)
        center_row_grid, center_col_grid = torch.meshgrid(
            center_rows, center_cols, indexing="ij"
        )
        center_coordinates = torch.stack(
            (center_row_grid.reshape(-1), center_col_grid.reshape(-1)), dim=-1
        )

        pixel_rows = torch.arange(height, device=device, dtype=torch.float32)
        pixel_cols = torch.arange(width, device=device, dtype=torch.float32)
        nearest_rows = torch.topk(
            (pixel_rows[:, None] - center_rows[None, :]).abs(),
            k=3,
            dim=1,
            largest=False,
        ).indices
        nearest_cols = torch.topk(
            (pixel_cols[:, None] - center_cols[None, :]).abs(),
            k=3,
            dim=1,
            largest=False,
        ).indices
        row_index = nearest_rows[:, None, :, None].expand(height, width, 3, 3)
        col_index = nearest_cols[None, :, None, :].expand(height, width, 3, 3)
        assignment_indices = (row_index * self.grid_cols + col_index).reshape(-1, 9)

        center_sample_rows = center_rows.round().long()
        center_sample_cols = center_cols.round().long()
        sample_row_grid, sample_col_grid = torch.meshgrid(
            center_sample_rows, center_sample_cols, indexing="ij"
        )
        center_sample_indices = (
            sample_row_grid.reshape(-1) * width + sample_col_grid.reshape(-1)
        )
        result = (assignment_indices, center_coordinates, center_sample_indices)
        self._spatial_cache[key] = result
        return result

    def _center_samples(
        self,
        values: torch.Tensor,
        sample_indices: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if self.center_pool_size == 1:
            return values[:, sample_indices, :]
        batch, _, channels = values.shape
        image = values.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        padding = self.center_pool_size // 2
        pooled = F.avg_pool2d(
            F.pad(image, (padding, padding, padding, padding), mode="replicate"),
            kernel_size=self.center_pool_size,
            stride=1,
        )
        pooled_values = pooled.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        return pooled_values[:, sample_indices, :]

    def _roi_center_sample_indices(
        self,
        roi_pixel_indices: torch.Tensor,
        roi_assignment_indices: torch.Tensor,
        roi_pixel_coordinates: torch.Tensor,
        center_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Select ROI-only prototypes without changing the assignment topology."""
        candidate_coordinates = center_coordinates[roi_assignment_indices]
        distances = (
            roi_pixel_coordinates[:, None, :] - candidate_coordinates
        ).square().sum(dim=-1)
        flat_tokens = roi_assignment_indices.reshape(-1)
        flat_distances = distances.reshape(-1)
        minimum_distances = distances.new_full((self.num_tokens,), float("inf"))
        minimum_distances.scatter_reduce_(
            0, flat_tokens, flat_distances, reduce="amin", include_self=True
        )
        local_pixels = torch.arange(
            roi_pixel_indices.numel(), device=roi_pixel_indices.device
        )[:, None].expand_as(roi_assignment_indices).reshape(-1)
        is_nearest = flat_distances <= minimum_distances[flat_tokens] + 1e-7
        fallback = int(roi_pixel_indices.numel())
        selected_local = torch.full(
            (self.num_tokens,),
            fallback,
            dtype=torch.long,
            device=roi_pixel_indices.device,
        )
        selected_local.scatter_reduce_(
            0,
            flat_tokens,
            torch.where(is_nearest, local_pixels, fallback),
            reduce="amin",
            include_self=True,
        )
        selected_local.clamp_max_(fallback - 1)
        return roi_pixel_indices[selected_local]

    def _prototype_samples(
        self,
        values: torch.Tensor,
        sample_indices: torch.Tensor,
        height: int,
        width: int,
        *,
        roi_active: bool,
    ) -> torch.Tensor:
        if roi_active:
            return values[:, sample_indices, :]
        return self._center_samples(values, sample_indices, height, width)

    def _token_spectral_means(
        self,
        image: torch.Tensor,
        assignment_weights: torch.Tensor,
        assignment_indices: torch.Tensor,
        token_mass: torch.Tensor,
        roi_pixel_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, _, _ = image.shape
        pixel_spectra = image.detach().permute(0, 2, 3, 1).reshape(batch, -1, channels)
        if roi_pixel_indices is not None:
            pixel_spectra = pixel_spectra[:, roi_pixel_indices, :]
        token_sum = torch.zeros(
            (batch, self.num_tokens, channels),
            dtype=torch.float32,
            device=image.device,
        )
        with torch.no_grad():
            weights = assignment_weights.detach().float()
            for start in range(0, pixel_spectra.shape[1], self.chunk_size):
                end = min(start + self.chunk_size, pixel_spectra.shape[1])
                local_spectra = pixel_spectra[:, start:end, :].float()
                local_indices = assignment_indices[start:end]
                for candidate in range(local_indices.shape[1]):
                    scatter_index = local_indices[:, candidate][None, :, None].expand(
                        batch, -1, channels
                    )
                    weighted_spectra = (
                        weights[:, start:end, candidate, None] * local_spectra
                    )
                    token_sum.scatter_add_(1, scatter_index, weighted_spectra)
            return token_sum / token_mass.detach().clamp_min(1e-6)[..., None]

    def _extra_context(
        self,
        pixel_spectrum: torch.Tensor,
        sample_indices: torch.Tensor,
        height: int,
        width: int,
    ) -> object | None:
        del pixel_spectrum, sample_indices, height, width
        return None

    def _extra_distance(
        self,
        extra_context: object | None,
        start: int,
        end: int,
        local_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        del extra_context, start, end, local_indices
        return None

    def _assignment_from_prototypes(
        self,
        pixel_spectrum: torch.Tensor,
        pixel_features: torch.Tensor,
        spectrum_prototypes: torch.Tensor,
        feature_prototypes: torch.Tensor,
        indices: torch.Tensor,
        center_coordinates: torch.Tensor,
        pixel_coordinates: torch.Tensor,
        spatial_scale: torch.Tensor,
        extra_context: object | None,
    ) -> torch.Tensor:
        assignment_parts: list[torch.Tensor] = []
        num_pixels = pixel_features.shape[1]
        for start in range(0, num_pixels, self.chunk_size):
            end = min(start + self.chunk_size, num_pixels)
            local_indices = indices[start:end]
            candidate_coordinates = center_coordinates[local_indices]
            spatial_delta = (
                pixel_coordinates[start:end, None, :] - candidate_coordinates
            ) / spatial_scale
            spatial_distance = spatial_delta.square().sum(dim=-1)[None, :, :]

            local_spectrum = pixel_spectrum[:, start:end, :]
            local_features = pixel_features[:, start:end, :]
            spectral_distance_parts: list[torch.Tensor] = []
            feature_distance_parts: list[torch.Tensor] = []
            for candidate in range(local_indices.shape[1]):
                candidate_indices = local_indices[:, candidate]
                spectrum_delta = (
                    local_spectrum - spectrum_prototypes[:, candidate_indices, :]
                )
                spectral_distance_parts.append(spectrum_delta.square().mean(dim=-1))
                feature_delta = (
                    local_features - feature_prototypes[:, candidate_indices, :]
                )
                feature_distance_parts.append(feature_delta.square().mean(dim=-1))
            spectral_distance = torch.stack(spectral_distance_parts, dim=-1)
            feature_distance = torch.stack(feature_distance_parts, dim=-1)
            distance = (
                self.spatial_weight * spatial_distance
                + self.spectral_weight * spectral_distance
                + self.feature_weight * feature_distance
            )
            extra_distance = self._extra_distance(
                extra_context,
                start,
                end,
                local_indices,
            )
            if extra_distance is not None:
                distance = distance + extra_distance
            assignment_parts.append(
                torch.softmax(-distance / self.temperature, dim=-1)
            )
        return torch.cat(assignment_parts, dim=1)

    def forward(
        self,
        spectrum: torch.Tensor,
        features: torch.Tensor,
        roi_pixel_indices: torch.Tensor | None = None,
    ) -> RegionOutput:
        if spectrum.ndim != 4 or features.ndim != 4:
            raise ValueError("spectrum and features must have shape [B, C, H, W]")
        if spectrum.shape[0] != features.shape[0] or spectrum.shape[2:] != features.shape[2:]:
            raise ValueError("spectrum and features must share batch and spatial dimensions")
        batch, _, height, width = spectrum.shape
        feature_dim = features.shape[1]
        device = features.device
        indices, center_coordinates, sample_indices = self._spatial_structure(
            height, width, device
        )
        full_pixel_count = height * width
        active_roi_indices = None
        pixel_rows = torch.arange(height, device=device, dtype=torch.float32)
        pixel_cols = torch.arange(width, device=device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(pixel_rows, pixel_cols, indexing="ij")
        full_pixel_coordinates = torch.stack(
            (row_grid.reshape(-1), col_grid.reshape(-1)), dim=-1
        )
        if roi_pixel_indices is not None:
            active_roi_indices = roi_pixel_indices.to(device=device, dtype=torch.long).reshape(-1)
            if active_roi_indices.numel() == 0:
                raise ValueError("roi_pixel_indices must select at least one pixel")
            if int(active_roi_indices.min().item()) < 0 or int(active_roi_indices.max().item()) >= full_pixel_count:
                raise ValueError("roi_pixel_indices contains out-of-range pixels")
            indices = indices[active_roi_indices]
            sample_indices = self._roi_center_sample_indices(
                active_roi_indices,
                indices,
                full_pixel_coordinates[active_roi_indices],
                center_coordinates,
            )

        pixel_spectrum = spectrum.permute(0, 2, 3, 1).reshape(batch, -1, spectrum.shape[1])
        pixel_features = features.permute(0, 2, 3, 1).reshape(batch, -1, feature_dim)
        extra_context = self._extra_context(
            pixel_spectrum,
            sample_indices,
            height,
            width,
        )
        pixel_coordinates = full_pixel_coordinates
        if active_roi_indices is not None:
            pixel_spectrum = pixel_spectrum[:, active_roi_indices, :]
            pixel_features = pixel_features[:, active_roi_indices, :]
            pixel_coordinates = pixel_coordinates[active_roi_indices]
        spatial_scale = torch.tensor(
            [max(height - 1, 1), max(width - 1, 1)], device=device, dtype=torch.float32
        )

        num_pixels = pixel_features.shape[1]
        center_spectrum = self._prototype_samples(
            spectrum.permute(0, 2, 3, 1).reshape(batch, -1, spectrum.shape[1]),
            sample_indices,
            height,
            width,
            roi_active=active_roi_indices is not None,
        )
        center_features = self._prototype_samples(
            features.permute(0, 2, 3, 1).reshape(batch, -1, feature_dim),
            sample_indices,
            height,
            width,
            roi_active=active_roi_indices is not None,
        )
        assignment = self._assignment_from_prototypes(
            pixel_spectrum,
            pixel_features,
            center_spectrum,
            center_features,
            indices,
            center_coordinates,
            pixel_coordinates,
            spatial_scale,
            extra_context,
        )
        token_sum = torch.zeros(
            (batch, self.num_tokens, feature_dim),
            dtype=torch.float32,
            device=device,
        )
        token_mass = torch.zeros(
            (batch, self.num_tokens),
            dtype=torch.float32,
            device=device,
        )
        for start in range(0, num_pixels, self.chunk_size):
            end = min(start + self.chunk_size, num_pixels)
            local_features = pixel_features[:, start:end, :].float()
            local_weights = assignment[:, start:end, :].float()
            local_indices = indices[start:end]
            for candidate in range(local_indices.shape[1]):
                scatter_index = local_indices[:, candidate][None, :].expand(batch, -1)
                weighted_features = (
                    local_weights[:, :, candidate, None] * local_features
                )
                token_sum.scatter_add_(
                    1,
                    scatter_index[:, :, None].expand(-1, -1, feature_dim),
                    weighted_features,
                )
                token_mass.scatter_add_(
                    1,
                    scatter_index,
                    local_weights[:, :, candidate],
                )
        token_features = token_sum / token_mass.clamp_min(self.empty_mass_threshold)[..., None]
        token_spectra = (
            self._token_spectral_means(
                spectrum,
                assignment,
                indices,
                token_mass,
                roi_pixel_indices=active_roi_indices,
            )
            if self.compute_token_spectra
            else None
        )
        empty_token_count = int((token_mass <= self.empty_mass_threshold).sum().item())
        entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(dim=-1).mean()
        normalized_center_coordinates = center_coordinates / spatial_scale
        return RegionOutput(
            assignment_weights=assignment,
            assignment_indices=indices,
            token_features=token_features,
            token_spectra=token_spectra,
            token_mass=token_mass,
            empty_token_count=empty_token_count,
            mean_assignment_entropy=entropy,
            token_coordinates=normalized_center_coordinates[None].expand(
                batch, -1, -1
            ),
            roi_pixel_indices=active_roi_indices,
            full_pixel_count=full_pixel_count,
        )
