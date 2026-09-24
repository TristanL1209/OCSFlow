"""OCSFlow network assembled from the production implementation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from .boundary_candidate_reweighter import BoundaryCandidateReweighter
from .encoder import LightweightPixelEncoder
from .graph_builder import GraphOutput, SpatialKNNGraphBuilder
from .regionizer import GridSoftRegionizer, RegionOutput
from .sparse_remote_token_coupling import (
    SparseRemoteTokenCoupling,
    SparseRemoteTokenCouplingSelection,
)
from .vector_fields import (
    LearnableScaleResidualTransformerGraphVectorField,
    TokenTargetOutput,
    build_train_token_targets,
)


@dataclass(frozen=True)
class GraphFMOutput:
    pixel_probabilities: torch.Tensor | None
    selected_pixel_probabilities: torch.Tensor | None
    selected_pixel_labels: torch.Tensor | None
    token_probabilities: torch.Tensor
    coarse_logits: torch.Tensor
    initial_state: torch.Tensor
    refined_logits: torch.Tensor
    used_noise: torch.Tensor
    assignment_weights: torch.Tensor
    assignment_indices: torch.Tensor
    token_features: torch.Tensor
    token_spectra: torch.Tensor | None
    token_mass: torch.Tensor
    region_output: RegionOutput
    graph_output: GraphOutput
    token_targets: TokenTargetOutput | None
    remote_token_coupling: SparseRemoteTokenCouplingSelection | None
    empty_token_count: int
    mean_assignment_entropy: torch.Tensor

    @property
    def refined_state(self) -> torch.Tensor:
        return self.refined_logits


class OCSFlow(nn.Module):
    """The complete OCSFlow model used in the paper."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        feature_dim: int,
        group_norm_groups: int,
        grid_rows: int,
        grid_cols: int,
        temperature: float,
        spatial_weight: float,
        spectral_weight: float,
        feature_weight: float,
        center_pool_size: int,
        chunk_size: int,
        k_neighbors: int,
        knn_chunk_size: int,
        graph_layers: int,
        fm_hidden_dim: int,
        sigma: float,
        euler_steps: int,
        graph_alpha: float,
        graph_beta: float,
        graph_gamma: float,
        residual_scale_init: float,
        transformer_attention: str,
        boundary: dict[str, object],
        remote: dict[str, object],
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.euler_steps = int(euler_steps)

        self.pixel_encoder = LightweightPixelEncoder(
            in_channels=in_channels,
            hidden_channels=feature_dim,
            groups=group_norm_groups,
        )
        self.token_regionizer = GridSoftRegionizer(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            temperature=temperature,
            spatial_weight=spatial_weight,
            spectral_weight=spectral_weight,
            feature_weight=feature_weight,
            center_pool_size=center_pool_size,
            chunk_size=chunk_size,
            compute_token_spectra=True,
        )
        self.graph_builder = SpatialKNNGraphBuilder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            k_neighbors=k_neighbors,
            knn_chunk_size=knn_chunk_size,
            num_layers=graph_layers,
            graph_alpha=graph_alpha,
            graph_beta=graph_beta,
            graph_gamma=graph_gamma,
        )
        self.vector_field = LearnableScaleResidualTransformerGraphVectorField(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=fm_hidden_dim,
            sigma=sigma,
            euler_steps=euler_steps,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            residual_scale_init=residual_scale_init,
            attention_mode=transformer_attention,
        )
        self.boundary_candidate_reweighter = BoundaryCandidateReweighter(
            pixel_feature_dim=feature_dim,
            token_feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=int(boundary["hidden_dim"]),
            correction_scale=float(boundary["correction_scale"]),
            detach_gate=bool(boundary["detach_gate"]),
            detach_reweight_inputs=bool(boundary["detach_reweight_inputs"]),
            chunk_size=chunk_size,
        )
        self.sparse_remote_token_coupling = SparseRemoteTokenCoupling(
            num_classes=num_classes,
            feature_dim=feature_dim,
            candidate_pool_multiplier=float(remote["candidate_pool_multiplier"]),
            remote_neighbors_per_anchor=int(remote["remote_neighbors_per_anchor"]),
            max_remote_reuse=int(remote["max_remote_reuse"]),
            similarity_temperature=float(remote["similarity_temperature"]),
            min_spatial_distance_factor=float(remote["min_spatial_distance_factor"]),
            detach_selection_features=bool(remote["detach_selection_features"]),
            similarity_chunk_size=int(remote.get("similarity_chunk_size", 1024)),
        )
        self._active_remote_token_coupling: SparseRemoteTokenCouplingSelection | None = None
        self.vector_field.register_forward_hook(
            self._apply_remote_token_coupling_hook,
            with_kwargs=True,
        )

    @property
    def nfe(self) -> int:
        return self.euler_steps

    def _apply_remote_token_coupling_hook(
        self,
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        base_velocity: torch.Tensor,
    ) -> torch.Tensor:
        del module, kwargs
        if self._active_remote_token_coupling is None:
            return base_velocity
        state, token_features = args[0], args[2]
        if not isinstance(state, torch.Tensor) or not isinstance(token_features, torch.Tensor):
            raise TypeError("vector-field state and token features must be tensors")
        return self.sparse_remote_token_coupling.apply_messages(
            base_velocity,
            state,
            token_features,
            self._active_remote_token_coupling,
        )

    @contextmanager
    def _remote_context(self, selection: SparseRemoteTokenCouplingSelection | None):
        previous = self._active_remote_token_coupling
        self._active_remote_token_coupling = selection
        try:
            yield
        finally:
            self._active_remote_token_coupling = previous

    @staticmethod
    def _selected_labels(
        ground_truth: torch.Tensor,
        mask: torch.Tensor,
        *,
        batch: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = ground_truth
        selected_mask = mask.bool()
        if labels.ndim == 2:
            labels = labels.unsqueeze(0).expand(batch, -1, -1)
            selected_mask = selected_mask.unsqueeze(0).expand(batch, -1, -1)
        if labels.shape != (batch, height, width) or selected_mask.shape != labels.shape:
            raise ValueError("ground_truth/mask must be [H, W] or [B, H, W]")
        selected = labels[selected_mask]
        if selected.numel() == 0:
            raise ValueError("classification mask does not select any pixels")
        return selected, selected_mask

    def forward(
        self,
        image: torch.Tensor,
        noise: torch.Tensor | None = None,
        ground_truth: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        remote_sampling_generator: torch.Generator | None = None,
    ) -> GraphFMOutput:
        if (ground_truth is None) != (train_mask is None):
            raise ValueError("ground_truth and train_mask must be provided together")
        batch, _, height, width = image.shape
        pixel_features = self.pixel_encoder(image)
        region = self.token_regionizer(image, pixel_features)
        graph = self.graph_builder(region)
        coarse_logits = self.graph_builder.classify(region, graph)
        initial_state, used_noise = self.vector_field.noisy_initial_state(coarse_logits, noise)

        targets = None
        if ground_truth is not None:
            targets = build_train_token_targets(
                region.assignment_weights,
                region.assignment_indices,
                ground_truth,
                train_mask,
                num_tokens=int(region.token_features.shape[1]),
                num_classes=self.num_classes,
            )
            observed_distribution = targets.class_proportions
            observation_strength = targets.observation_strength
        else:
            token_count = int(region.token_features.shape[1])
            observed_distribution = image.new_zeros((batch, token_count, self.num_classes))
            observation_strength = image.new_zeros((batch, token_count))

        coupling = None
        if train_mask is not None:
            coupling = self.sparse_remote_token_coupling.select(
                region.assignment_weights,
                region.assignment_indices,
                train_mask,
                region.token_features,
                self.graph_builder.center_coordinates,
                sampling_generator=remote_sampling_generator,
            )

        with self._remote_context(coupling):
            refined_logits = self.vector_field.integrate(
                initial_state,
                region,
                graph,
                observed_token_mask=observation_strength > 0.0,
                observed_class_distribution=observed_distribution,
                observation_strength=observation_strength,
            )
        token_probabilities = torch.softmax(refined_logits, dim=-1)

        selected_probabilities = None
        selected_labels = None
        normalized_train_mask = None
        if self.training and ground_truth is not None and train_mask is not None:
            selected_labels, normalized_train_mask = self._selected_labels(
                ground_truth,
                train_mask,
                batch=batch,
                height=height,
                width=width,
            )

        if normalized_train_mask is not None:
            reweighted = self.boundary_candidate_reweighter.forward_indexed_selected(
                pixel_features,
                region.token_features,
                token_probabilities,
                region.assignment_weights,
                region.assignment_indices,
                normalized_train_mask,
            )
            selected_probabilities = reweighted.final_pixel_probabilities
            pixel_probabilities = None
        else:
            reweighted = self.boundary_candidate_reweighter.forward_indexed(
                pixel_features,
                region.token_features,
                token_probabilities,
                region.assignment_weights,
                region.assignment_indices,
            )
            pixel_probabilities = reweighted.final_pixel_probabilities.reshape(
                batch, height, width, self.num_classes
            ).permute(0, 3, 1, 2)

        return GraphFMOutput(
            pixel_probabilities=pixel_probabilities,
            selected_pixel_probabilities=selected_probabilities,
            selected_pixel_labels=selected_labels,
            token_probabilities=token_probabilities,
            coarse_logits=coarse_logits,
            initial_state=initial_state,
            refined_logits=refined_logits,
            used_noise=used_noise,
            assignment_weights=region.assignment_weights,
            assignment_indices=region.assignment_indices,
            token_features=region.token_features,
            token_spectra=region.token_spectra,
            token_mass=region.token_mass,
            region_output=region,
            graph_output=graph,
            token_targets=targets,
            remote_token_coupling=coupling,
            empty_token_count=region.empty_token_count,
            mean_assignment_entropy=region.mean_assignment_entropy,
        )

    def training_flow_matching_loss(
        self,
        output: GraphFMOutput,
        ground_truth: torch.Tensor,
        train_mask: torch.Tensor,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TokenTargetOutput, torch.Tensor]:
        del ground_truth, train_mask
        targets = output.token_targets
        if targets is None:
            raise ValueError("training output does not contain token targets")
        with self._remote_context(output.remote_token_coupling):
            loss, sampled_time = self.vector_field.flow_matching_loss(
                output.initial_state,
                targets.target_state,
                targets.observed_token_mask,
                output.region_output,
                output.graph_output,
                observed_token_mask=targets.observation_strength > 0.0,
                observed_class_distribution=targets.class_proportions,
                observation_strength=targets.observation_strength,
                time=time,
            )
        return loss, targets, sampled_time
