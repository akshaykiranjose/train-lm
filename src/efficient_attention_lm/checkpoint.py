from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def capture_rng_state(sampler_state: dict | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": sampler_state,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # CPU RNG state must be a CPU ByteTensor, even when the
    # checkpoint was loaded with map_location="cuda".
    torch.set_rng_state(state["torch"].cpu())

    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in state["cuda"]]
        )


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    config: dict,
    sampler_state: dict | None = None,
    best_validation_loss: float | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "config": config,
        "rng_state": capture_rng_state(sampler_state),
        "best_validation_loss": best_validation_loss,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
