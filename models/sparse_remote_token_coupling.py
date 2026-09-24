"""Label-free sparse remote-token messages for supervised token objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SparseRemoteTokenCouplingSelection:
    """Detached remote neighborhoods indexed by directly supervised anchors."""

    anchor_mask: torch.Tensor
    candidate_pool_mask: torch.Tensor
    remote_neighbor_indices: torch.Tensor
    remote_neighbor_mask: torch.Tensor
    attention_weights: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class SparseRemoteTokenCoupling(nn.Module):
    """Add label-free remote state messages to directly supervised velocities."""

    def __init__(
        self,
        *,
        num_classes: int,
        feature_dim: int,
        candidate_pool_multiplier: float = 4.0,
        remote_neighbors_per_anchor: int = 2,
        max_remote_reuse: int = 4,
        similarity_temperature: float = 0.10,
        min_spatial_distance_factor: float = 2.0,
        detach_selection_features: bool = True,
        similarity_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if candidate_pool_multiplier < 1:
            raise ValueError("candidate_pool_multiplier must be at least 1")
        if remote_neighbors_per_anchor < 1:
            raise ValueError("remote_neighbors_per_anchor must be positive")
        if max_remote_reuse < 1:
            raise ValueError("max_remote_reuse must be positive")
        if similarity_temperature <= 0:
            raise ValueError("similarity_temperature must be positive")
        if min_spatial_distance_factor < 0:
            raise ValueError("min_spatial_distance_factor must be non-negative")
        if not detach_selection_features:
            raise ValueError("remote selection features must remain detached")
        if similarity_chunk_size < 1:
            raise ValueError("similarity_chunk_size must be positive")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.candidate_pool_multiplier = float(candidate_pool_multiplier)
        self.remote_neighbors_per_anchor = int(remote_neighbors_per_anchor)
        self.max_remote_reuse = int(max_remote_reuse)
        self.similarity_temperature = float(similarity_temperature)
        self.min_spatial_distance_factor = float(min_spatial_distance_factor)
        self.detach_selection_features = bool(detach_selection_features)
        self.similarity_chunk_size = int(similarity_chunk_size)

        message_dim = self.feature_dim
        self.message_function = nn.Sequential(
            nn.Linear(self.num_classes + self.feature_dim, message_dim),
            nn.GELU(),
        )
        self.message_gate = nn.Linear(
            self.num_classes + self.feature_dim,
            1,
        )
        self.message_output = nn.Linear(message_dim, self.num_classes)
        nn.init.zeros_(self.message_output.weight)
        nn.init.zeros_(self.message_output.bias)

    @staticmethod
    def _resolve_sampling_generator(
        reference: torch.Tensor,
        sampling_generator: torch.Generator | None,
    ) -> torch.Generator:
        if sampling_generator is not None:
            return sampling_generator
        fallback = torch.Generator(device=reference.device)
        fallback.manual_seed(0)
        return fallback

    @staticmethod
    def _batched_mask(
        train_mask: torch.Tensor,
        batch: int,
        pixel_count: int,
    ) -> torch.Tensor:
        if train_mask.numel() == pixel_count and batch == 1:
            return train_mask.reshape(1, pixel_count).bool()
        if train_mask.numel() == batch * pixel_count:
            return train_mask.reshape(batch, pixel_count).bool()
        raise ValueError("train_mask does not match the assignment pixel count")

    @staticmethod
    def _coordinates_for_batch(
        token_coordinates: torch.Tensor,
        batch: int,
        num_tokens: int,
    ) -> torch.Tensor:
        if token_coordinates.ndim == 2:
            coordinates = token_coordinates[None].expand(batch, -1, -1)
        elif token_coordinates.ndim == 3 and token_coordinates.shape[0] == batch:
            coordinates = token_coordinates
        else:
            raise ValueError("token_coordinates must have shape [M, 2] or [B, M, 2]")
        if coordinates.shape[1] != num_tokens:
            raise ValueError("token coordinate count must match num_tokens")
        return coordinates

    @staticmethod
    def _coordinate_stride(coordinates: torch.Tensor) -> torch.Tensor:
        positive_steps: list[torch.Tensor] = []
        for axis in range(coordinates.shape[-1]):
            unique = torch.unique(coordinates[:, axis]).sort().values
            if unique.numel() > 1:
                differences = unique[1:] - unique[:-1]
                positive_steps.append(differences[differences > 0].min())
        if not positive_steps:
            return coordinates.new_tensor(1.0)
        return torch.stack(positive_steps).min()

    def direct_supervised_mask(
        self,
        assignment_weights: torch.Tensor,
        assignment_indices: torch.Tensor,
        train_mask: torch.Tensor,
        *,
        num_tokens: int,
    ) -> torch.Tensor:
        """Return tokens receiving at least one train pixel's hard top-1 assignment."""
        if assignment_weights.ndim != 3:
            raise ValueError("assignment_weights must have shape [B, N, candidates]")
        batch, pixel_count, candidates = assignment_weights.shape
        if assignment_indices.shape != (pixel_count, candidates):
            raise ValueError("assignment_indices must have shape [N, candidates]")
        mask = self._batched_mask(train_mask, batch, pixel_count)
        with torch.no_grad():
            weights = assignment_weights.detach()
            indices = assignment_indices.to(device=weights.device, dtype=torch.long)
            top1_position = weights.argmax(dim=-1)
            top1_token = indices[None].expand(batch, -1, -1).gather(
                -1,
                top1_position[..., None],
            ).squeeze(-1)
            direct_count = torch.zeros(
                (batch, num_tokens),
                dtype=torch.long,
                device=weights.device,
            )
            direct_count.scatter_add_(1, top1_token, mask.to(torch.long))
        return direct_count > 0

    def _sample_candidate_pool(
        self,
        unlabeled: torch.Tensor,
        anchor_count: int,
        sampling_generator: torch.Generator,
    ) -> torch.Tensor:
        pool_size = min(
            int(unlabeled.numel()),
            max(0, round(float(anchor_count) * self.candidate_pool_multiplier)),
        )
        if pool_size == 0:
            return unlabeled[:0]
        order = torch.randperm(
            unlabeled.numel(),
            device=unlabeled.device,
            generator=sampling_generator,
        )[:pool_size]
        return unlabeled[order]

    @torch.no_grad()
    def select(
        self,
        assignment_weights: torch.Tensor,
        assignment_indices: torch.Tensor,
        train_mask: torch.Tensor,
        token_features: torch.Tensor,
        token_coordinates: torch.Tensor,
        *,
        sampling_generator: torch.Generator | None = None,
    ) -> SparseRemoteTokenCouplingSelection:
        """Select remote neighbors without reading labels or differentiable features."""
        if token_features.ndim != 3:
            raise ValueError("token_features must have shape [B, M, D]")
        batch, num_tokens, feature_dim = token_features.shape
        if feature_dim != self.feature_dim:
            raise ValueError("token feature dimension does not match coupling module")
        if assignment_weights.shape[0] != batch:
            raise ValueError("assignment and token feature batch sizes must match")
        generator = self._resolve_sampling_generator(token_features, sampling_generator)
        coordinates = self._coordinates_for_batch(
            token_coordinates,
            batch,
            num_tokens,
        )
        anchor_mask = self.direct_supervised_mask(
            assignment_weights,
            assignment_indices,
            train_mask,
            num_tokens=num_tokens,
        )
        candidate_pool_mask = torch.zeros_like(anchor_mask)
        neighbor_indices = torch.zeros(
            (batch, num_tokens, self.remote_neighbors_per_anchor),
            dtype=torch.long,
            device=token_features.device,
        )
        neighbor_mask = torch.zeros_like(neighbor_indices, dtype=torch.bool)
        attention_weights = token_features.new_zeros(neighbor_indices.shape)

        unlabeled_count = 0
        candidate_pool_size = 0
        selected_edge_count = 0
        selected_unique_remote_count = 0
        similarity_pair_count = 0
        full_search_pair_count = 0
        max_observed_reuse = 0

        selection_features = token_features.detach()
        for batch_index in range(batch):
            anchors = torch.nonzero(anchor_mask[batch_index], as_tuple=False).flatten()
            unlabeled = torch.nonzero(~anchor_mask[batch_index], as_tuple=False).flatten()
            anchor_count = int(anchors.numel())
            unlabeled_count += int(unlabeled.numel())
            full_search_pair_count += int(unlabeled.numel() * anchors.numel())
            if anchor_count == 0 or unlabeled.numel() == 0:
                continue
            candidate_pool = self._sample_candidate_pool(
                unlabeled,
                anchor_count,
                generator,
            )
            pool_size = int(candidate_pool.numel())
            if pool_size == 0:
                continue
            candidate_pool_mask[batch_index, candidate_pool] = True
            candidate_pool_size += pool_size
            similarity_pair_count += pool_size * anchor_count

            anchor_features = F.normalize(
                selection_features[batch_index, anchors],
                dim=-1,
            )
            similarities = selection_features.new_empty((pool_size, anchor_count))
            spatial_valid = torch.empty(
                (pool_size, anchor_count),
                dtype=torch.bool,
                device=token_features.device,
            )
            minimum_distance = (
                self.min_spatial_distance_factor
                * self._coordinate_stride(coordinates[batch_index])
            )
            for start in range(0, pool_size, self.similarity_chunk_size):
                end = min(start + self.similarity_chunk_size, pool_size)
                local_pool = candidate_pool[start:end]
                remote_features = F.normalize(
                    selection_features[batch_index, local_pool],
                    dim=-1,
                )
                similarities[start:end] = remote_features @ anchor_features.transpose(0, 1)
                distances = torch.cdist(
                    coordinates[batch_index, local_pool],
                    coordinates[batch_index, anchors],
                )
                spatial_valid[start:end] = distances >= minimum_distance

            ranked_device = torch.argsort(
                similarities.transpose(0, 1).masked_fill(
                    ~spatial_valid.transpose(0, 1),
                    -torch.inf,
                ),
                dim=-1,
                descending=True,
            )
            ranked_valid_device = spatial_valid.transpose(0, 1).gather(
                1,
                ranked_device,
            )
            anchor_order_device = torch.randperm(
                anchor_count,
                device=token_features.device,
                generator=generator,
            )
            ranked = ranked_device.cpu()
            ranked_valid = ranked_valid_device.cpu()
            anchor_order = anchor_order_device.cpu()
            reuse_count = torch.zeros(pool_size, dtype=torch.long)
            selected_pool_positions: list[list[int]] = [
                [] for _ in range(anchor_count)
            ]
            rank_cursor = [0 for _ in range(anchor_count)]
            ordered_anchors = anchor_order.tolist()
            for neighbor_round in range(self.remote_neighbors_per_anchor):
                round_order = (
                    ordered_anchors[neighbor_round:]
                    + ordered_anchors[:neighbor_round]
                )
                for anchor_position in round_order:
                    while rank_cursor[anchor_position] < pool_size:
                        rank_position = rank_cursor[anchor_position]
                        rank_cursor[anchor_position] += 1
                        if not bool(ranked_valid[anchor_position, rank_position]):
                            break
                        pool_position = int(ranked[anchor_position, rank_position])
                        if int(reuse_count[pool_position]) >= self.max_remote_reuse:
                            continue
                        selected_pool_positions[anchor_position].append(pool_position)
                        reuse_count[pool_position] += 1
                        break

            selected_positions_cpu = torch.tensor(
                [
                    pool_positions
                    + [0] * (self.remote_neighbors_per_anchor - len(pool_positions))
                    for pool_positions in selected_pool_positions
                ],
                dtype=torch.long,
            )
            selected_mask_cpu = torch.tensor(
                [
                    [True] * len(pool_positions)
                    + [False]
                    * (self.remote_neighbors_per_anchor - len(pool_positions))
                    for pool_positions in selected_pool_positions
                ],
                dtype=torch.bool,
            )
            active_anchor_positions_cpu = torch.nonzero(
                selected_mask_cpu.any(dim=1),
                as_tuple=False,
            ).flatten()
            if active_anchor_positions_cpu.numel() > 0:
                active_anchor_positions = active_anchor_positions_cpu.to(
                    device=token_features.device,
                )
                selected_positions = selected_positions_cpu.index_select(
                    0,
                    active_anchor_positions_cpu,
                ).to(device=token_features.device)
                selected_mask = selected_mask_cpu.index_select(
                    0,
                    active_anchor_positions_cpu,
                ).to(device=token_features.device)
                active_anchor_indices = anchors.index_select(
                    0,
                    active_anchor_positions,
                )
                selected_indices = candidate_pool[selected_positions]
                selected_indices = selected_indices.masked_fill(~selected_mask, 0)
                selected_similarities = similarities.transpose(0, 1).index_select(
                    0,
                    active_anchor_positions,
                ).gather(1, selected_positions)
                selected_attention = torch.softmax(
                    selected_similarities.masked_fill(~selected_mask, -torch.inf)
                    / self.similarity_temperature,
                    dim=1,
                ).masked_fill(~selected_mask, 0.0)
                neighbor_indices[
                    batch_index,
                    active_anchor_indices,
                ] = selected_indices
                neighbor_mask[
                    batch_index,
                    active_anchor_indices,
                ] = selected_mask
                attention_weights[
                    batch_index,
                    active_anchor_indices,
                ] = selected_attention
            selected_edge_count += int(reuse_count.sum())
            selected_unique_remote_count += int((reuse_count > 0).sum())
            max_observed_reuse = max(
                max_observed_reuse,
                int(reuse_count.max().item()) if reuse_count.numel() else 0,
            )

        pair_reduction = (
            1.0 - float(similarity_pair_count) / float(full_search_pair_count)
            if full_search_pair_count > 0
            else 0.0
        )
        diagnostics = {
            "direct_supervised_token_count": anchor_mask.sum().detach(),
            "unlabeled_token_count": token_features.new_tensor(unlabeled_count),
            "candidate_pool_size": token_features.new_tensor(candidate_pool_size),
            "candidate_pool_ratio": token_features.new_tensor(
                float(candidate_pool_size) / float(unlabeled_count)
                if unlabeled_count > 0
                else 0.0
            ),
            "similarity_pair_count": token_features.new_tensor(similarity_pair_count),
            "full_search_pair_count": token_features.new_tensor(full_search_pair_count),
            "pair_reduction_ratio": token_features.new_tensor(pair_reduction),
            "selected_remote_edge_count": token_features.new_tensor(selected_edge_count),
            "selected_unique_remote_token_count": token_features.new_tensor(
                selected_unique_remote_count
            ),
            "max_observed_remote_reuse": token_features.new_tensor(max_observed_reuse),
        }
        return SparseRemoteTokenCouplingSelection(
            anchor_mask=anchor_mask.detach(),
            candidate_pool_mask=candidate_pool_mask.detach(),
            remote_neighbor_indices=neighbor_indices.detach(),
            remote_neighbor_mask=neighbor_mask.detach(),
            attention_weights=attention_weights.detach(),
            diagnostics=diagnostics,
        )

    def apply_messages(
        self,
        base_velocity: torch.Tensor,
        state: torch.Tensor,
        token_features: torch.Tensor,
        selection: SparseRemoteTokenCouplingSelection | None,
    ) -> torch.Tensor:
        """Apply one remote message residual without another vector-field call."""
        if selection is None:
            return base_velocity
        indices = selection.remote_neighbor_indices.to(device=state.device)
        valid = selection.remote_neighbor_mask.to(device=state.device)
        weights = selection.attention_weights.to(dtype=state.dtype, device=state.device)
        batch_indices = torch.arange(state.shape[0], device=state.device)[:, None, None]
        remote_state = state[batch_indices, indices]
        remote_features = token_features[batch_indices, indices]
        remote_message = self.message_function(
            torch.cat((remote_state, remote_features), dim=-1)
        )
        aggregate = (
            remote_message
            * weights[..., None]
            * valid[..., None].to(remote_message.dtype)
        ).sum(dim=2)
        gate = torch.sigmoid(
            self.message_gate(torch.cat((state, token_features), dim=-1))
        )
        delta = gate * self.message_output(aggregate)
        delta = delta * selection.anchor_mask[..., None].to(
            dtype=delta.dtype,
            device=delta.device,
        )
        return base_velocity + delta
