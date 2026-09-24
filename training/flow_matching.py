"""OCSFlow pixel-classification and logits flow-matching losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.model import GraphFMOutput, OCSFlow


def masked_probability_cross_entropy(
    probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError("probabilities must be [B,K,H,W]")
    batch, classes, height, width = probabilities.shape
    labels = ground_truth
    selected_mask = mask.bool()
    if labels.ndim == 2:
        labels = labels.unsqueeze(0).expand(batch, -1, -1)
        selected_mask = selected_mask.unsqueeze(0).expand(batch, -1, -1)
    if labels.shape != (batch, height, width) or selected_mask.shape != labels.shape:
        raise ValueError("ground_truth/mask shape mismatch")
    selected = probabilities.permute(0, 2, 3, 1)[selected_mask]
    targets = labels[selected_mask] - 1
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= classes:
        raise ValueError("mask must select raw labels in [1,K]")
    return F.nll_loss(selected.clamp_min(eps).log(), targets)


def selected_probability_cross_entropy(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    targets = labels.reshape(-1) - 1
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.numel():
        raise ValueError("selected probabilities and labels do not align")
    return F.nll_loss(probabilities.clamp_min(eps).log(), targets)


def make_evaluation_noise(model: OCSFlow, reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (reference.shape[0], model.graph_builder.token_count, model.num_classes),
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )


def classification_and_flow_loss(
    model: OCSFlow,
    output: GraphFMOutput,
    ground_truth: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    fm_weight: float,
    classification_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fm_loss, _, sampled_time = model.training_flow_matching_loss(
        output,
        ground_truth,
        train_mask,
    )
    if output.selected_pixel_probabilities is None or output.selected_pixel_labels is None:
        raise RuntimeError("formal training forward must return selected pixel probabilities")
    classification_loss = selected_probability_cross_entropy(
        output.selected_pixel_probabilities,
        output.selected_pixel_labels,
    )
    total = float(fm_weight) * fm_loss + float(classification_weight) * classification_loss
    return total, fm_loss, classification_loss, sampled_time
