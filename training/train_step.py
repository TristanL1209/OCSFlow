"""One whole-scene OCSFlow optimization step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.model import OCSFlow
from .flow_matching import classification_and_flow_loss
from .seed import assert_finite


@dataclass(frozen=True)
class TrainStepResult:
    total_loss: float
    flow_matching_loss: float
    classification_loss: float
    sampled_time: float
    labeled_token_count: int


def train_step(
    model: OCSFlow,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    fm_weight: float,
    classification_weight: float,
    remote_sampling_generator: torch.Generator,
) -> TrainStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        image,
        ground_truth=ground_truth,
        train_mask=train_mask,
        remote_sampling_generator=remote_sampling_generator,
    )
    total, fm_loss, classification_loss, sampled_time = classification_and_flow_loss(
        model,
        output,
        ground_truth,
        train_mask,
        fm_weight=fm_weight,
        classification_weight=classification_weight,
    )
    for name, value in (("FM loss", fm_loss), ("classification loss", classification_loss),
                        ("total loss", total)):
        assert_finite(name, value)
    total.backward()
    optimizer.step()
    if output.token_targets is None:
        raise RuntimeError("supervised forward did not create token targets")
    return TrainStepResult(
        total_loss=float(total.detach()),
        flow_matching_loss=float(fm_loss.detach()),
        classification_loss=float(classification_loss.detach()),
        sampled_time=float(sampled_time.detach().mean()),
        labeled_token_count=int(output.token_targets.observed_token_mask.sum().detach()),
    )
