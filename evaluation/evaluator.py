"""Whole-scene validation and test evaluation."""

from __future__ import annotations

from typing import Any

import torch

from models.model import OCSFlow
from training.flow_matching import masked_probability_cross_entropy
from .metrics import classification_metrics


@torch.no_grad()
def evaluate_split(
    model: OCSFlow,
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    train_mask: torch.Tensor,
    evaluation_mask: torch.Tensor,
    *,
    noise: torch.Tensor,
    remote_sampling_generator: torch.Generator,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    output = model(
        image,
        noise=noise,
        ground_truth=ground_truth,
        train_mask=train_mask,
        remote_sampling_generator=remote_sampling_generator,
    )
    if output.pixel_probabilities is None:
        raise RuntimeError("evaluation forward did not return full pixel probabilities")
    loss = masked_probability_cross_entropy(
        output.pixel_probabilities,
        ground_truth,
        evaluation_mask,
    )
    prediction = output.pixel_probabilities.argmax(dim=1)[0] + 1
    metrics = classification_metrics(
        ground_truth[evaluation_mask].detach().cpu().numpy(),
        prediction[evaluation_mask].detach().cpu().numpy(),
        model.num_classes,
    )
    return float(loss), metrics


def evaluate_model(
    model: OCSFlow,
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    noise: torch.Tensor,
    remote_sampling_generator: torch.Generator,
) -> dict[str, Any]:
    loss, metrics = evaluate_split(
        model, image, ground_truth, train_mask, test_mask,
        noise=noise, remote_sampling_generator=remote_sampling_generator,
    )
    return {"test_loss": loss, **metrics}
