"""Lightweight spectral--spatial encoder used by OCSFlow."""

from __future__ import annotations

import torch
from torch import nn


def _valid_group_count(channels: int, requested_groups: int) -> int:
    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _valid_group_count(channels, groups)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.depthwise_norm = nn.GroupNorm(norm_groups, channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pointwise_norm = nn.GroupNorm(norm_groups, channels)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        features = self.activation(self.depthwise_norm(self.depthwise(inputs)))
        features = self.pointwise_norm(self.pointwise(features))
        return self.activation(features + residual)


class LightweightPixelEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _valid_group_count(hidden_channels, groups)
        self.spectral_projection = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.projection_norm = nn.GroupNorm(norm_groups, hidden_channels)
        self.activation = nn.GELU()
        self.spatial_blocks = nn.Sequential(
            DepthwiseSeparableResidualBlock(hidden_channels, groups),
            DepthwiseSeparableResidualBlock(hidden_channels, groups),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.projection_norm(self.spectral_projection(inputs)))
        return self.spatial_blocks(features)
