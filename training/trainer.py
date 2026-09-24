"""OCSFlow epoch loop with validation-selected best-state restoration."""

from __future__ import annotations

from typing import Any

import torch

from evaluation.evaluator import evaluate_split
from models.model import OCSFlow
from .checkpoint import clone_state_dict, restore_state_dict
from .flow_matching import make_evaluation_noise
from .seed import assert_finite, set_reproducible_seed
from .train_step import train_step


def train_model(
    model: OCSFlow,
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    seed = int(config["seed"])
    set_reproducible_seed(seed)
    assert_finite("input image", image)
    model.to(image.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training["min_learning_rate"]),
    )
    evaluation_noise = make_evaluation_noise(
        model,
        image,
        seed + int(training["evaluation_noise_seed_offset"]),
    )
    sampling_generator = torch.Generator(device=image.device)
    sampling_generator.manual_seed(seed + 7919)
    best_state: dict[str, torch.Tensor] | None = None
    best_oa = -1.0
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    interval = int(training["log_interval"])
    for epoch in range(1, epochs + 1):
        step = train_step(
            model,
            optimizer,
            image,
            ground_truth,
            train_mask,
            fm_weight=float(training["fm_loss_weight"]),
            classification_weight=float(training["classification_loss_weight"]),
            remote_sampling_generator=sampling_generator,
        )
        scheduler.step()
        val_loss, metrics = evaluate_split(
            model,
            image,
            ground_truth,
            train_mask,
            val_mask,
            noise=evaluation_noise,
            remote_sampling_generator=sampling_generator,
        )
        oa = float(metrics["oa"])
        history.append({"epoch": epoch, "train_loss": step.total_loss,
                        "classification_loss": step.classification_loss,
                        "flow_matching_loss": step.flow_matching_loss,
                        "validation_loss": val_loss, "validation_oa": oa})
        if oa > best_oa or (oa == best_oa and val_loss < best_loss):
            best_oa, best_loss, best_epoch = oa, val_loss, epoch
            best_state = clone_state_dict(model)
        if epoch == 1 or epoch % interval == 0 or epoch == epochs:
            print(f"epoch={epoch}/{epochs} total={step.total_loss:.4f} "
                  f"fm={step.flow_matching_loss:.4f} cls={step.classification_loss:.4f} "
                  f"val_oa={100.0 * oa:.2f}%")
    if best_state is None:
        raise RuntimeError("training did not produce a validation-selected state")
    restore_state_dict(model, best_state)
    return {"best_epoch": best_epoch, "best_validation_oa": best_oa,
            "best_validation_loss": best_loss, "history": history,
            "evaluation_noise": evaluation_noise}
