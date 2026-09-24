"""Best-state cloning and restoration."""

from __future__ import annotations

import torch
from torch import nn


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def restore_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)
