"""Semantic-disagreement-gated reweighting of local token candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn



def normalize_semantic_disagreement_gate(
    disagreement: torch.Tensor,
    *,
    gate_quantile: float = 0.95,
    detach_gate: bool = True,
) -> torch.Tensor:
    """Normalize Jensen-Shannon disagreement into a stable [0, 1] gate."""
    if disagreement.ndim != 3 or disagreement.shape[-1] != 1:
        raise ValueError("disagreement must have shape [B, N, 1]")
    if not 0.0 < gate_quantile <= 1.0:
        raise ValueError("gate_quantile must be in (0, 1]")
    scale = torch.quantile(
        disagreement.detach().squeeze(-1).float(),
        float(gate_quantile),
        dim=1,
        keepdim=True,
    ).to(dtype=disagreement.dtype)
    gate = (disagreement / (scale[..., None] + 1e-8)).clamp(0.0, 1.0)
    gate = torch.nan_to_num(gate)
    return gate.detach() if detach_gate else gate


def semantic_disagreement(
    candidate_probabilities: torch.Tensor,
    assignment_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted Jensen-Shannon disagreement for local token candidates."""
    if candidate_probabilities.ndim != 4:
        raise ValueError(
            "candidate_probabilities must have shape [B, N, candidates, C]"
        )
    if assignment_weights.shape != candidate_probabilities.shape[:-1]:
        raise ValueError(
            "assignment_weights must match candidate probability axes"
        )
    weights = assignment_weights[..., None]
    mean_probability = (weights * candidate_probabilities).sum(dim=2)
    log_ratio = torch.log(candidate_probabilities + 1e-8) - torch.log(
        mean_probability[:, :, None, :] + 1e-8
    )
    disagreement = (
        weights * candidate_probabilities * log_ratio
    ).sum(dim=(2, 3), keepdim=False)[..., None]
    return torch.nan_to_num(disagreement).clamp_min(0.0)


@dataclass(frozen=True)
class BoundaryCandidateReweightOutput:
    final_pixel_logits: torch.Tensor
    final_pixel_probabilities: torch.Tensor
    corrected_weights: torch.Tensor
    delta: torch.Tensor
    centered_correction: torch.Tensor
    semantic_disagreement: torch.Tensor
    boundary_gate: torch.Tensor


@dataclass(frozen=True)
class BoundaryCandidateReweightSelectedOutput:
    final_pixel_logits: torch.Tensor
    final_pixel_probabilities: torch.Tensor
    corrected_weights: torch.Tensor
    delta: torch.Tensor
    centered_correction: torch.Tensor
    semantic_disagreement: torch.Tensor
    boundary_gate: torch.Tensor


class BoundaryCandidateReweighter(nn.Module):
    """Adjust only the nine token-to-pixel mixing weights at each pixel."""

    def __init__(
        self,
        pixel_feature_dim: int,
        token_feature_dim: int,
        num_classes: int,
        hidden_dim: int = 32,
        correction_scale: float = 1.0,
        detach_gate: bool = True,
        detach_reweight_inputs: bool = True,
        chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        if min(pixel_feature_dim, token_feature_dim, num_classes, hidden_dim) <= 0:
            raise ValueError("feature dimensions, num_classes and hidden_dim must be positive")
        if correction_scale < 0:
            raise ValueError("correction_scale must be non-negative")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.pixel_feature_dim = int(pixel_feature_dim)
        self.token_feature_dim = int(token_feature_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.correction_scale = float(correction_scale)
        self.detach_gate = bool(detach_gate)
        self.detach_reweight_inputs = bool(detach_reweight_inputs)
        self.chunk_size = int(chunk_size)
        self.gate_quantile = 0.95
        self.eps = 1e-8

        self.pixel_projection = nn.Sequential(
            nn.Linear(self.pixel_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.token_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.compatibility = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.final_linear = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        self.last_diagnostics: dict[str, float] = {}
        self.last_pixel_features_shape: tuple[int, ...] | None = None
        self.last_candidate_token_features_shape: tuple[int, ...] | None = None
        self.last_candidate_token_logits_shape: tuple[int, ...] | None = None
        self.last_candidate_token_probabilities_shape: tuple[int, ...] | None = None

    @staticmethod
    def _flatten_pixel_features(
        pixel_features: torch.Tensor,
        pixel_count: int,
    ) -> torch.Tensor:
        if pixel_features.ndim == 4:
            batch, channels, height, width = pixel_features.shape
            if height * width != pixel_count:
                raise ValueError("pixel feature spatial shape does not match assignment")
            return pixel_features.permute(0, 2, 3, 1).reshape(
                batch, pixel_count, channels
            )
        if pixel_features.ndim == 3 and pixel_features.shape[1] == pixel_count:
            return pixel_features
        raise ValueError("pixel_features must have shape [B, D, H, W] or [B, N, D]")

    def semantic_gate(
        self,
        candidate_token_logits: torch.Tensor,
        assignment_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_probabilities = torch.softmax(candidate_token_logits, dim=-1)
        disagreement = semantic_disagreement(
            candidate_probabilities,
            assignment_weights,
        )
        gate = normalize_semantic_disagreement_gate(
            disagreement,
            gate_quantile=self.gate_quantile,
            detach_gate=self.detach_gate,
        )
        return disagreement, gate

    def apply_assignment_correction(
        self,
        assignment_weights: torch.Tensor,
        delta: torch.Tensor,
        semantic_disagreement_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if assignment_weights.shape != delta.shape:
            raise ValueError("assignment_weights and delta must have the same shape")
        if semantic_disagreement_gate.shape != (*delta.shape[:-1], 1):
            raise ValueError("gate must have shape [B, N, 1]")
        bounded = torch.tanh(delta)
        centered = bounded - (
            assignment_weights * bounded
        ).sum(dim=-1, keepdim=True)
        base_logits = torch.log(
            assignment_weights.clamp_min(torch.finfo(assignment_weights.dtype).tiny)
        )
        base_weights = torch.softmax(base_logits, dim=-1)
        shifted_weights = torch.softmax(
            base_logits
            + self.correction_scale * semantic_disagreement_gate * centered,
            dim=-1,
        )
        corrected_weights = (
            assignment_weights + (shifted_weights - base_weights)
        ).clamp_min(0.0)
        return corrected_weights, centered

    def _record_diagnostics(
        self,
        output: BoundaryCandidateReweightOutput | BoundaryCandidateReweightSelectedOutput,
        assignment_weights: torch.Tensor,
        *,
        pixel_shape: tuple[int, ...],
        candidate_feature_shape: tuple[int, ...],
        candidate_probability_shape: tuple[int, ...],
    ) -> None:
        with torch.no_grad():
            original = assignment_weights.detach()
            corrected = output.corrected_weights.detach()
            entropy_before = -(
                original * original.clamp_min(self.eps).log()
            ).sum(dim=-1)
            entropy_after = -(
                corrected * corrected.clamp_min(self.eps).log()
            ).sum(dim=-1)
            self.last_diagnostics = {
                "gate_mean": float(output.boundary_gate.detach().float().mean().cpu()),
                "weight_change_abs_mean": float(
                    (corrected - original).abs().float().mean().cpu()
                ),
                "top1_switch_rate": float(
                    (original.argmax(dim=-1) != corrected.argmax(dim=-1))
                    .float()
                    .mean()
                    .cpu()
                ),
                "entropy_before": float(entropy_before.float().mean().cpu()),
                "entropy_after": float(entropy_after.float().mean().cpu()),
                "correction_abs_mean": float(
                    output.centered_correction.detach().abs().float().mean().cpu()
                ),
            }
        self.last_pixel_features_shape = pixel_shape
        self.last_candidate_token_features_shape = candidate_feature_shape
        self.last_candidate_token_probabilities_shape = candidate_probability_shape

    @staticmethod
    def _gather_candidates(
        token_values: torch.Tensor,
        flattened_indices: torch.Tensor,
        *,
        pixel_count: int,
        candidate_count: int,
    ) -> torch.Tensor:
        gathered = token_values.index_select(1, flattened_indices)
        return gathered.reshape(
            token_values.shape[0],
            pixel_count,
            candidate_count,
            token_values.shape[-1],
        )

    @staticmethod
    def _selected_mask_2d(
        selected_pixel_mask: torch.Tensor,
        *,
        batch: int,
        pixel_count: int,
    ) -> torch.Tensor:
        mask = selected_pixel_mask.to(dtype=torch.bool)
        if mask.ndim == 2:
            if mask.numel() != pixel_count:
                raise ValueError("selected_pixel_mask spatial size does not match pixels")
            mask = mask.reshape(1, pixel_count).expand(batch, -1)
        elif mask.ndim == 3:
            if mask.shape[0] != batch or mask[0].numel() != pixel_count:
                raise ValueError("selected_pixel_mask must match [B, H, W]")
            mask = mask.reshape(batch, pixel_count)
        else:
            raise ValueError("selected_pixel_mask must have shape [H, W] or [B, H, W]")
        if not bool(mask.any()):
            raise ValueError("selected_pixel_mask does not select any pixels")
        return mask

    def _indexed_semantic_disagreement(
        self,
        token_probabilities: torch.Tensor,
        assignment_weights: torch.Tensor,
        assignment_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Exact full-scene gate statistic without retaining candidate probabilities."""
        _, pixel_count, candidate_count = assignment_weights.shape
        flattened_indices = assignment_indices.reshape(-1)
        gate_probabilities = (
            token_probabilities.detach() if self.detach_gate else token_probabilities
        )
        gate_weights = (
            assignment_weights.detach() if self.detach_gate else assignment_weights
        )
        disagreement_chunks: list[torch.Tensor] = []
        for start in range(0, pixel_count, self.chunk_size):
            end = min(start + self.chunk_size, pixel_count)
            local_indices = flattened_indices[
                start * candidate_count : end * candidate_count
            ]
            local_probabilities = self._gather_candidates(
                gate_probabilities,
                local_indices,
                pixel_count=end - start,
                candidate_count=candidate_count,
            )
            disagreement_chunks.append(
                semantic_disagreement(
                    local_probabilities,
                    gate_weights[:, start:end],
                )
            )
        return torch.cat(disagreement_chunks, dim=1)

    def _compatibility_delta(
        self,
        pixel_embeddings: torch.Tensor,
        candidate_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if pixel_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError("pixel embedding dimension does not match hidden_dim")
        if candidate_token_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError("token embedding dimension does not match hidden_dim")
        if candidate_token_embeddings.shape[:-2] != pixel_embeddings.shape[:-1]:
            raise ValueError("pixel/token embedding leading dimensions do not match")
        linear = self.compatibility[0]
        weight_pixel, weight_token, weight_difference = linear.weight.split(
            self.hidden_dim,
            dim=1,
        )
        expanded_pixels = pixel_embeddings.unsqueeze(-2)
        hidden = (
            F.linear(pixel_embeddings, weight_pixel).unsqueeze(-2)
            + F.linear(candidate_token_embeddings, weight_token)
            + F.linear(
                (expanded_pixels - candidate_token_embeddings).abs(),
                weight_difference,
                linear.bias,
            )
        )
        hidden = self.compatibility[2](self.compatibility[1](hidden))
        return self.final_linear(hidden).squeeze(-1)

    def forward(
        self,
        pixel_features: torch.Tensor,
        candidate_token_features: torch.Tensor,
        candidate_token_logits: torch.Tensor,
        assignment_weights: torch.Tensor,
        semantic_disagreement_gate: torch.Tensor,
        *,
        record_diagnostics: bool | None = None,
    ) -> BoundaryCandidateReweightOutput:
        if assignment_weights.ndim != 3:
            raise ValueError("assignment_weights must have shape [B, N, candidates]")
        batch, pixel_count, candidate_count = assignment_weights.shape
        pixels = self._flatten_pixel_features(pixel_features, pixel_count)
        if pixels.shape != (batch, pixel_count, self.pixel_feature_dim):
            raise ValueError("pixel feature dimension does not match reweighter")
        expected_token_shape = (
            batch,
            pixel_count,
            candidate_count,
            self.token_feature_dim,
        )
        if candidate_token_features.shape != expected_token_shape:
            raise ValueError("candidate_token_features has an unexpected shape")
        expected_logits_shape = (
            batch,
            pixel_count,
            candidate_count,
            self.num_classes,
        )
        if candidate_token_logits.shape != expected_logits_shape:
            raise ValueError("candidate_token_logits has an unexpected shape")
        if semantic_disagreement_gate.shape != (batch, pixel_count, 1):
            raise ValueError("semantic_disagreement_gate must have shape [B, N, 1]")

        compatibility_pixels = pixels.detach() if self.detach_reweight_inputs else pixels
        compatibility_tokens = (
            candidate_token_features.detach()
            if self.detach_reweight_inputs
            else candidate_token_features
        )
        pixel_embedding = self.pixel_projection(compatibility_pixels)
        token_embedding = self.token_projection(compatibility_tokens)
        delta = self._compatibility_delta(pixel_embedding, token_embedding)
        corrected_weights, centered = self.apply_assignment_correction(
            assignment_weights,
            delta,
            semantic_disagreement_gate,
        )
        candidate_probabilities = torch.softmax(candidate_token_logits, dim=-1)
        final_probabilities = (
            corrected_weights[..., None] * candidate_probabilities
        ).sum(dim=2)
        final_logits = torch.log(final_probabilities.clamp_min(self.eps))
        disagreement = semantic_disagreement(
            candidate_probabilities,
            assignment_weights,
        )
        output = BoundaryCandidateReweightOutput(
            final_pixel_logits=final_logits,
            final_pixel_probabilities=final_probabilities,
            corrected_weights=corrected_weights,
            delta=delta,
            centered_correction=centered,
            semantic_disagreement=disagreement,
            boundary_gate=semantic_disagreement_gate,
        )
        should_record = (not self.training) if record_diagnostics is None else bool(
            record_diagnostics
        )
        if should_record:
            self.last_candidate_token_logits_shape = tuple(
                candidate_token_logits.shape
            )
            self._record_diagnostics(
                output,
                assignment_weights,
                pixel_shape=tuple(pixels.shape),
                candidate_feature_shape=tuple(candidate_token_features.shape),
                candidate_probability_shape=tuple(candidate_probabilities.shape),
            )
        return output

    def forward_indexed(
        self,
        pixel_features: torch.Tensor,
        token_features: torch.Tensor,
        token_probabilities: torch.Tensor,
        assignment_weights: torch.Tensor,
        assignment_indices: torch.Tensor,
        *,
        record_diagnostics: bool | None = None,
    ) -> BoundaryCandidateReweightOutput:
        """Project global features once, then gather nine candidates in chunks."""
        if assignment_weights.ndim != 3 or assignment_indices.ndim != 2:
            raise ValueError("assignment weights/indices have unexpected ranks")
        batch, pixel_count, candidate_count = assignment_weights.shape
        if assignment_indices.shape != (pixel_count, candidate_count):
            raise ValueError("assignment_indices shape does not match weights")
        if token_features.ndim != 3 or token_probabilities.ndim != 3:
            raise ValueError("token features/probabilities must have shape [B, M, D]")
        if token_features.shape[:2] != token_probabilities.shape[:2]:
            raise ValueError("token features and probabilities must share [B, M]")
        if token_features.shape[0] != batch:
            raise ValueError("token and assignment batch sizes differ")
        if token_features.shape[-1] != self.token_feature_dim:
            raise ValueError("token feature dimension does not match reweighter")
        if token_probabilities.shape[-1] != self.num_classes:
            raise ValueError("token probability class dimension does not match")
        pixels = self._flatten_pixel_features(pixel_features, pixel_count)
        if pixels.shape != (batch, pixel_count, self.pixel_feature_dim):
            raise ValueError("pixel feature dimension does not match reweighter")

        compatibility_pixels = pixels.detach() if self.detach_reweight_inputs else pixels
        compatibility_tokens = (
            token_features.detach() if self.detach_reweight_inputs else token_features
        )
        pixel_embeddings = self.pixel_projection(compatibility_pixels)
        token_embeddings = self.token_projection(compatibility_tokens)
        flattened_indices = assignment_indices.reshape(-1)
        chunk_ranges = tuple(
            (start, min(start + self.chunk_size, pixel_count))
            for start in range(0, pixel_count, self.chunk_size)
        )

        disagreement = self._indexed_semantic_disagreement(
            token_probabilities,
            assignment_weights,
            assignment_indices,
        )
        gate = normalize_semantic_disagreement_gate(
            disagreement,
            gate_quantile=self.gate_quantile,
            detach_gate=self.detach_gate,
        )

        final_logits_chunks: list[torch.Tensor] = []
        final_probability_chunks: list[torch.Tensor] = []
        corrected_weight_chunks: list[torch.Tensor] = []
        delta_chunks: list[torch.Tensor] = []
        centered_chunks: list[torch.Tensor] = []
        for start, end in chunk_ranges:
            local_flattened_indices = flattened_indices[
                start * candidate_count : end * candidate_count
            ]
            local_probabilities = self._gather_candidates(
                token_probabilities,
                local_flattened_indices,
                pixel_count=end - start,
                candidate_count=candidate_count,
            )
            candidate_embeddings = self._gather_candidates(
                token_embeddings,
                local_flattened_indices,
                pixel_count=end - start,
                candidate_count=candidate_count,
            )
            delta = self._compatibility_delta(
                pixel_embeddings[:, start:end],
                candidate_embeddings,
            )
            corrected_weights, centered = self.apply_assignment_correction(
                assignment_weights[:, start:end],
                delta,
                gate[:, start:end],
            )
            final_probabilities = (
                corrected_weights[..., None]
                * local_probabilities
            ).sum(dim=2)
            final_logits_chunks.append(
                torch.log(final_probabilities.clamp_min(self.eps))
            )
            final_probability_chunks.append(final_probabilities)
            corrected_weight_chunks.append(corrected_weights)
            delta_chunks.append(delta)
            centered_chunks.append(centered)
        output = BoundaryCandidateReweightOutput(
            final_pixel_logits=torch.cat(final_logits_chunks, dim=1),
            final_pixel_probabilities=torch.cat(final_probability_chunks, dim=1),
            corrected_weights=torch.cat(corrected_weight_chunks, dim=1),
            delta=torch.cat(delta_chunks, dim=1),
            centered_correction=torch.cat(centered_chunks, dim=1),
            semantic_disagreement=disagreement,
            boundary_gate=gate,
        )
        should_record = (not self.training) if record_diagnostics is None else bool(
            record_diagnostics
        )
        if should_record:
            self.last_candidate_token_logits_shape = None
            self._record_diagnostics(
                output,
                assignment_weights,
                pixel_shape=tuple(pixels.shape),
                candidate_feature_shape=(
                    batch,
                    pixel_count,
                    candidate_count,
                    int(token_features.shape[-1]),
                ),
                candidate_probability_shape=(
                    batch,
                    pixel_count,
                    candidate_count,
                    int(token_probabilities.shape[-1]),
                ),
            )
        return output

    def forward_indexed_selected(
        self,
        pixel_features: torch.Tensor,
        token_features: torch.Tensor,
        token_probabilities: torch.Tensor,
        assignment_weights: torch.Tensor,
        assignment_indices: torch.Tensor,
        selected_pixel_mask: torch.Tensor,
        *,
        record_diagnostics: bool | None = None,
    ) -> BoundaryCandidateReweightSelectedOutput:
        """Reweight only selected pixels while retaining the exact full-scene gate."""
        if assignment_weights.ndim != 3 or assignment_indices.ndim != 2:
            raise ValueError("assignment weights/indices have unexpected ranks")
        batch, pixel_count, candidate_count = assignment_weights.shape
        if assignment_indices.shape != (pixel_count, candidate_count):
            raise ValueError("assignment_indices shape does not match weights")
        if token_features.ndim != 3 or token_probabilities.ndim != 3:
            raise ValueError("token features/probabilities must have shape [B, M, D]")
        if token_features.shape[:2] != token_probabilities.shape[:2]:
            raise ValueError("token features and probabilities must share [B, M]")
        if token_features.shape[0] != batch:
            raise ValueError("token and assignment batch sizes differ")
        if token_features.shape[-1] != self.token_feature_dim:
            raise ValueError("token feature dimension does not match reweighter")
        if token_probabilities.shape[-1] != self.num_classes:
            raise ValueError("token probability class dimension does not match")
        pixels = self._flatten_pixel_features(pixel_features, pixel_count)
        if pixels.shape != (batch, pixel_count, self.pixel_feature_dim):
            raise ValueError("pixel feature dimension does not match reweighter")

        selected_mask = self._selected_mask_2d(
            selected_pixel_mask,
            batch=batch,
            pixel_count=pixel_count,
        )
        selected_batch, selected_pixel = selected_mask.nonzero(as_tuple=True)
        selected_candidate_indices = assignment_indices.index_select(
            0,
            selected_pixel,
        )
        selected_weights = assignment_weights[selected_batch, selected_pixel]

        disagreement = self._indexed_semantic_disagreement(
            token_probabilities,
            assignment_weights,
            assignment_indices,
        )
        full_gate = normalize_semantic_disagreement_gate(
            disagreement,
            gate_quantile=self.gate_quantile,
            detach_gate=self.detach_gate,
        )
        selected_gate = full_gate[selected_batch, selected_pixel]

        selected_pixels = pixels[selected_batch, selected_pixel]
        selected_token_features = token_features[
            selected_batch[:, None],
            selected_candidate_indices,
        ]
        compatibility_pixels = (
            selected_pixels.detach() if self.detach_reweight_inputs else selected_pixels
        )
        compatibility_tokens = (
            selected_token_features.detach()
            if self.detach_reweight_inputs
            else selected_token_features
        )
        pixel_embeddings = self.pixel_projection(compatibility_pixels)
        candidate_embeddings = self.token_projection(compatibility_tokens)
        delta = self._compatibility_delta(pixel_embeddings, candidate_embeddings)
        corrected_weights, centered = self.apply_assignment_correction(
            selected_weights,
            delta,
            selected_gate,
        )
        selected_candidate_probabilities = token_probabilities[
            selected_batch[:, None],
            selected_candidate_indices,
        ]
        final_probabilities = (
            corrected_weights[..., None] * selected_candidate_probabilities
        ).sum(dim=1)
        output = BoundaryCandidateReweightSelectedOutput(
            final_pixel_logits=torch.log(final_probabilities.clamp_min(self.eps)),
            final_pixel_probabilities=final_probabilities,
            corrected_weights=corrected_weights,
            delta=delta,
            centered_correction=centered,
            semantic_disagreement=disagreement,
            boundary_gate=selected_gate,
        )
        should_record = (not self.training) if record_diagnostics is None else bool(
            record_diagnostics
        )
        if should_record:
            self.last_candidate_token_logits_shape = None
            self._record_diagnostics(
                output,
                selected_weights,
                pixel_shape=tuple(selected_pixels.shape),
                candidate_feature_shape=tuple(selected_token_features.shape),
                candidate_probability_shape=tuple(
                    selected_candidate_probabilities.shape
                ),
            )
        return output

    def diagnostics(self) -> dict[str, float]:
        return dict(self.last_diagnostics)


__all__ = [
    "BoundaryCandidateReweightOutput",
    "BoundaryCandidateReweightSelectedOutput",
    "BoundaryCandidateReweighter",
]
