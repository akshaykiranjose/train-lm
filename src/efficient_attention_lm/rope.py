from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0, initial_length: int = 256) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even")
        self.dim = dim
        self.theta = theta
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self._build_cache(initial_length, device=torch.device("cpu"))

    def _build_cache(self, length: int, device: torch.device) -> None:
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.cos_cache = angles.cos()
        self.sin_cache = angles.sin()

    def _ensure_cache(self, length: int, device: torch.device) -> None:
        if self.cos_cache.device != device or self.cos_cache.shape[0] < length:
            new_length = max(length, max(1, self.cos_cache.shape[0]) * 2)
            self._build_cache(new_length, device)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape or q.shape[-1] != self.dim:
            raise ValueError("q and k must have equal shape ending in the RoPE dimension")
        sequence_length = q.shape[-2]
        if positions is None:
            positions = torch.arange(sequence_length, device=q.device)
        positions = positions.to(device=q.device, dtype=torch.long)
        if positions.ndim != 1 or positions.numel() != sequence_length:
            raise ValueError("positions must be a one-dimensional tensor of sequence length")
        if bool((positions < 0).any()):
            raise ValueError("positions must be non-negative")
        self._ensure_cache(int(positions.max().item()) + 1, q.device)
        cos = self.cos_cache.index_select(0, positions).to(q.dtype)[None, None, :, :]
        sin = self.sin_cache.index_select(0, positions).to(q.dtype)[None, None, :, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)
