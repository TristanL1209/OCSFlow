"""Formal logits class-state flow, window Transformer and Euler solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..graph_builder import GraphOutput
from ..regionizer import RegionOutput


@dataclass(frozen=True)
class TokenTargetOutput:
    class_proportions: torch.Tensor
    target_state: torch.Tensor
    labeled_token_mask: torch.Tensor
    labeled_mass: torch.Tensor
    observation_strength: torch.Tensor
    label_mass_expected: torch.Tensor
    label_mass_actual: torch.Tensor
    label_mass_abs_error: torch.Tensor
    max_per_pixel_mass_error: torch.Tensor

    @property
    def observed_token_mask(self) -> torch.Tensor:
        return self.labeled_token_mask


def _centered_logits(distribution: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    normalized = distribution.clamp(min=eps, max=1.0)
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(eps)
    logits = normalized.log()
    return logits - logits.mean(dim=-1, keepdim=True)


def build_train_token_targets(assignment_weights: torch.Tensor,
                              assignment_indices: torch.Tensor,
                              ground_truth: torch.Tensor, train_mask: torch.Tensor,
                              num_tokens: int, num_classes: int,
                              eps: float = 1e-8,
                              roi_pixel_indices: torch.Tensor | None = None) -> TokenTargetOutput:
    """Route each labeled pixel to its strongest soft-supertoken candidate."""
    if assignment_weights.ndim != 3 or assignment_indices.shape != assignment_weights.shape[1:]:
        raise ValueError("invalid soft-supertoken assignment shapes")
    batch = assignment_weights.shape[0]
    if ground_truth.ndim == 2:
        ground_truth = ground_truth.unsqueeze(0).expand(batch, -1, -1)
        train_mask = train_mask.unsqueeze(0).expand(batch, -1, -1)
    flat_labels = ground_truth.reshape(batch, -1)
    flat_masks = train_mask.reshape(batch, -1)
    if roi_pixel_indices is not None:
        flat_labels = flat_labels[:, roi_pixel_indices]
        flat_masks = flat_masks[:, roi_pixel_indices]
    class_masses, expected, actual, pixel_errors = [], [], [], []
    for index in range(batch):
        labels = flat_labels[index, flat_masks[index]]
        weights = assignment_weights[index, flat_masks[index]].detach()
        indices = assignment_indices[flat_masks[index]]
        if labels.numel() == 0:
            raise ValueError("train_mask selects no labeled pixels")
        winners = weights.argmax(dim=-1, keepdim=True)
        selected = indices.gather(1, winners).reshape(-1)
        one_hot = F.one_hot(labels - 1, num_classes=num_classes).to(torch.float64)
        class_mass = torch.zeros((num_tokens, num_classes), dtype=torch.float64,
                                 device=assignment_weights.device)
        class_mass.index_add_(0, selected, one_hot)
        class_masses.append(class_mass.to(assignment_weights.dtype))
        expected.append(class_mass.new_tensor(float(labels.numel())))
        actual.append(class_mass.sum())
        pixel_errors.append(class_mass.new_zeros(()))
    class_mass = torch.stack(class_masses).detach()
    expected_mass = torch.stack(expected).detach()
    actual_mass = torch.stack(actual).detach()
    labeled_mass = class_mass.sum(dim=-1).detach()
    strength = labeled_mass.clamp(min=0.0, max=1.0)
    observed = strength > eps
    proportions = (class_mass / labeled_mass.clamp_min(eps)[..., None]).detach()
    endpoint = _centered_logits(proportions)
    target = torch.where(observed[..., None], endpoint, torch.zeros_like(endpoint))
    return TokenTargetOutput(proportions, target, observed, labeled_mass, strength,
                             expected_mass, actual_mass,
                             (actual_mass - expected_mass).abs(),
                             torch.stack(pixel_errors).detach())


class DeterministicGraphODEFunc(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.state_projection = nn.Linear(num_classes * 2, hidden_dim)
        self.time_projection = nn.Linear(1, hidden_dim, bias=False)
        self.derivative = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, num_classes))

    def forward(self, state: torch.Tensor, time: torch.Tensor,
                token_features: torch.Tensor, neighbor_indices: torch.Tensor,
                neighbor_weights: torch.Tensor,
                neighbor_mask: torch.Tensor | None = None) -> torch.Tensor:
        neighbor_state = state[:, neighbor_indices, :]
        weights = neighbor_weights[None] if neighbor_weights.ndim == 2 else neighbor_weights
        if neighbor_mask is not None:
            weights = weights * neighbor_mask.to(weights)[None]
        aggregated = (neighbor_state * weights[..., None]).sum(dim=2)
        time_value = time.reshape(1, 1, 1).to(state)
        hidden = (self.state_projection(torch.cat((state, aggregated), dim=-1))
                  + self.feature_projection(token_features)
                  + self.time_projection(time_value).expand(state.shape[0], state.shape[1], -1))
        return self.derivative(hidden)


class GraphVectorField(nn.Module):
    state_space = "logits"

    def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int = 64,
                 sigma: float = 0.2, euler_steps: int = 10) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.sigma = float(sigma)
        self.euler_steps = int(euler_steps)
        if self.sigma < 0 or self.euler_steps < 1:
            raise ValueError("sigma must be non-negative and euler_steps positive")

    @property
    def nfe(self) -> int:
        return self.euler_steps

    def noisy_initial_state(self, coarse_logits: torch.Tensor,
                            noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(coarse_logits)
        if noise.shape != coarse_logits.shape:
            raise ValueError("noise must match coarse logits")
        return coarse_logits + self.sigma * noise, noise

    def _conditioned_features(self, token_features: torch.Tensor,
                              observed_strength: torch.Tensor,
                              observed_distribution: torch.Tensor) -> torch.Tensor:
        strength = observed_strength.to(token_features).clamp(0.0, 1.0)
        distribution = observed_distribution.to(token_features) * strength[..., None]
        return torch.cat((token_features, strength[..., None], distribution), dim=-1)

    def flow_matching_loss(self, initial_state: torch.Tensor, target_state: torch.Tensor,
                           labeled_token_mask: torch.Tensor, region_output: RegionOutput,
                           graph_output: GraphOutput, observed_token_mask: torch.Tensor,
                           observed_class_distribution: torch.Tensor,
                           observation_strength: torch.Tensor,
                           time: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = initial_state.shape[0]
        if time is None:
            time = torch.rand((batch, 1, 1), device=initial_state.device,
                              dtype=initial_state.dtype)
        path_state = (1.0 - time) * initial_state + time * target_state
        target_velocity = target_state - initial_state
        predicted = self.forward(path_state, time.reshape(-1)[0],
                                 region_output.token_features, graph_output,
                                 observed_class_distribution, observation_strength)
        error = (predicted - target_velocity).square().mean(dim=-1)
        weights = observation_strength.to(error).clamp(0.0, 1.0)
        if not labeled_token_mask.any():
            raise ValueError("no token receives training-label mass")
        return (error * weights).sum() / weights.sum().clamp_min(1e-8), time

    def integrate(self, initial_state: torch.Tensor, region_output: RegionOutput,
                  graph_output: GraphOutput, observed_token_mask: torch.Tensor,
                  observed_class_distribution: torch.Tensor,
                  observation_strength: torch.Tensor) -> torch.Tensor:
        state = initial_state
        step_size = 1.0 / self.euler_steps
        for step in range(self.euler_steps):
            time = state.new_tensor(step * step_size)
            velocity = self.forward(state, time, region_output.token_features,
                                    graph_output, observed_class_distribution,
                                    observation_strength)
            state = state + step_size * velocity
        return state


class ShallowGraphVectorField(GraphVectorField):
    def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int = 64,
                 sigma: float = 0.2, euler_steps: int = 10) -> None:
        super().__init__(num_classes, feature_dim, hidden_dim, sigma, euler_steps)
        self.vector_field = DeterministicGraphODEFunc(
            num_classes, feature_dim + num_classes + 1, hidden_dim)

    def forward(self, state: torch.Tensor, time: torch.Tensor,
                token_features: torch.Tensor, graph_output: GraphOutput,
                observed_distribution: torch.Tensor,
                observed_strength: torch.Tensor) -> torch.Tensor:
        conditioned = self._conditioned_features(token_features, observed_strength,
                                                 observed_distribution)
        return self.vector_field(state, time, conditioned, graph_output.neighbor_indices,
                                 graph_output.neighbor_weights, graph_output.neighbor_mask)


class SinusoidalTimeMLP(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(),
                                 nn.Linear(model_dim, model_dim))

    def forward(self, time: torch.Tensor, *, batch_size: int, token_count: int,
                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        value = time.reshape(1).to(dtype=dtype, device=device)
        half = self.model_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=dtype,
                                                                 device=device) / max(half, 1))
        phases = value[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        if embedding.shape[-1] < self.model_dim:
            embedding = F.pad(embedding, (0, self.model_dim - embedding.shape[-1]))
        return self.mlp(embedding[:, :self.model_dim])[:, None, :].expand(
            batch_size, token_count, self.model_dim)


class SupertokenAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model_dim, self.num_heads = int(model_dim), int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    def _attention(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                  dropout_p=self.dropout if self.training else 0.0)
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, self.model_dim))

    @staticmethod
    def _partition(x: torch.Tensor, size: int) -> torch.Tensor:
        batch, height, width, channels = x.shape
        return x.view(batch, height // size, size, width // size, size, channels).permute(
            0, 1, 3, 2, 4, 5).reshape(-1, size * size, channels)

    @staticmethod
    def _reverse(windows: torch.Tensor, batch: int, height: int, width: int,
                 size: int) -> torch.Tensor:
        channels = windows.shape[-1]
        return windows.view(batch, height // size, width // size, size, size, channels).permute(
            0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)

    def forward(self, x: torch.Tensor, *, mode: str, grid_rows: int, grid_cols: int,
                window_size: int, shift_size: int) -> torch.Tensor:
        if mode == "global":
            return self._attention(x)
        batch, tokens, channels = x.shape
        image = x.reshape(batch, grid_rows, grid_cols, channels)
        pad_rows = (window_size - grid_rows % window_size) % window_size
        pad_cols = (window_size - grid_cols % window_size) % window_size
        padded = F.pad(image, (0, 0, 0, pad_cols, 0, pad_rows))
        height, width = grid_rows + pad_rows, grid_cols + pad_cols
        valid = F.pad(x.new_ones((batch, grid_rows, grid_cols, 1)),
                      (0, 0, 0, pad_cols, 0, pad_rows))
        if shift_size:
            padded = torch.roll(padded, (-shift_size, -shift_size), (1, 2))
            valid = torch.roll(valid, (-shift_size, -shift_size), (1, 2))
        windows, valid_windows = self._partition(padded, window_size), self._partition(valid, window_size)
        mask = torch.zeros((windows.shape[0], window_size ** 2, window_size ** 2),
                           dtype=x.dtype, device=x.device)
        mask = mask.masked_fill(valid_windows.squeeze(-1)[:, None, :] <= 0, -100.0)
        if shift_size:
            region = torch.zeros((1, height, width, 1), device=x.device)
            slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            count = 0
            for hs in slices:
                for ws in slices:
                    region[:, hs, ws, :] = count; count += 1
            region = self._partition(region, window_size).squeeze(-1)
            region = region[:, None, :] - region[:, :, None]
            region = region.to(x.dtype).masked_fill(region != 0, -100.0)
            mask = mask + region.repeat(batch, 1, 1)
        attended = self._reverse(self._attention(windows, mask[:, None]), batch,
                                 height, width, window_size)
        if shift_size:
            attended = torch.roll(attended, (shift_size, shift_size), (1, 2))
        return attended[:, :grid_rows, :grid_cols].reshape(batch, tokens, channels)


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1, self.attention = nn.LayerNorm(model_dim), SupertokenAttention(model_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, 2 * model_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * model_dim, model_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), **kwargs)
        return x + self.mlp(self.norm2(x))


class TransformerResidual(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, grid_rows: int, grid_cols: int,
                 attention_mode: str = "auto", model_dim: int = 96,
                 depth: int = 2, num_heads: int = 4, window_size: int = 8,
                 global_token_limit: int = 1024) -> None:
        super().__init__()
        self.num_classes, self.grid_rows, self.grid_cols = num_classes, grid_rows, grid_cols
        self.attention_mode, self.window_size = attention_mode, window_size
        self.global_token_limit = global_token_limit
        self.state_projection = nn.Linear(num_classes, model_dim)
        self.feature_projection = nn.Linear(feature_dim, model_dim)
        self.observation_projection = nn.Linear(num_classes + 1, model_dim)
        self.row_embedding, self.col_embedding = nn.Embedding(grid_rows, model_dim), nn.Embedding(grid_cols, model_dim)
        self.time_embedding = SinusoidalTimeMLP(model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(model_dim, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, num_classes)
        nn.init.zeros_(self.output_projection.weight); nn.init.zeros_(self.output_projection.bias)

    def forward(self, state: torch.Tensor, time: torch.Tensor, token_features: torch.Tensor,
                observed_distribution: torch.Tensor, observed_strength: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = state.shape
        rows = torch.arange(self.grid_rows, device=state.device)
        cols = torch.arange(self.grid_cols, device=state.device)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        position = self.row_embedding(row_grid.reshape(-1)) + self.col_embedding(col_grid.reshape(-1))
        observation = torch.cat((observed_strength[..., None],
                                 observed_distribution * observed_strength[..., None]), dim=-1)
        hidden = (self.state_projection(state) + self.feature_projection(token_features)
                  + self.observation_projection(observation) + position.to(state)[None]
                  + self.time_embedding(time, batch_size=batch, token_count=token_count,
                                        dtype=state.dtype, device=state.device))
        mode = self.attention_mode
        if mode == "auto":
            mode = "global" if token_count <= self.global_token_limit else "window"
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, mode=mode, grid_rows=self.grid_rows, grid_cols=self.grid_cols,
                           window_size=self.window_size,
                           shift_size=self.window_size // 2 if mode == "window" and index % 2 else 0)
        return self.output_projection(self.final_norm(hidden))


class LearnableScaleResidualTransformerGraphVectorField(ShallowGraphVectorField):
    def __init__(self, num_classes: int, feature_dim: int, hidden_dim: int,
                 sigma: float, euler_steps: int, grid_rows: int, grid_cols: int,
                 attention_mode: str = "auto", residual_scale_init: float = 0.1) -> None:
        super().__init__(num_classes, feature_dim, hidden_dim, sigma, euler_steps)
        self.transformer_branch = TransformerResidual(num_classes, feature_dim,
                                                       grid_rows, grid_cols, attention_mode)
        self.transformer_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, state: torch.Tensor, time: torch.Tensor,
                token_features: torch.Tensor, graph_output: GraphOutput,
                observed_distribution: torch.Tensor,
                observed_strength: torch.Tensor) -> torch.Tensor:
        base = super().forward(state, time, token_features, graph_output,
                               observed_distribution, observed_strength)
        delta = self.transformer_branch(state, time, token_features,
                                        observed_distribution, observed_strength)
        return base + delta * self.transformer_residual_scale.to(state)
