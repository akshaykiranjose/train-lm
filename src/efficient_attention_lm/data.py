from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class RandomWindowSampler:
    """Random independent next-token windows over a memory-mapped token stream."""

    def __init__(self, path: str | Path, context_length: int, seed: int) -> None:
        self.path = Path(path)
        self.context_length = int(context_length)
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= self.context_length:
            raise ValueError(
                f"{self.path} has {len(self.tokens)} tokens; need more than "
                f"context_length={self.context_length}"
            )
        self.rng = np.random.default_rng(seed)

    @property
    def num_valid_starts(self) -> int:
        return len(self.tokens) - self.context_length

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        starts: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if starts is None:
            starts = self.rng.integers(
                0, self.num_valid_starts, size=batch_size, dtype=np.int64
            )
        elif len(starts) != batch_size:
            raise ValueError("number of starts must equal batch_size")
        offsets = np.arange(self.context_length + 1, dtype=np.int64)
        windows = np.asarray(self.tokens[starts[:, None] + offsets[None, :]], dtype=np.int64)
        x = torch.from_numpy(windows[:, :-1])
        y = torch.from_numpy(windows[:, 1:])
        if device is not None:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        return x, y

    def fixed_starts(self, count: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.num_valid_starts, size=count, dtype=np.int64)

    def state_dict(self) -> dict:
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng_state"]

