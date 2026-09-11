from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from ..config import ModelConfig
from ..rope import RotaryEmbedding


class AttentionBase(nn.Module, ABC):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            initial_length=config.context_length,
        )

    def project_qkv(
        self, x: torch.Tensor, positions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = x.shape
        shape = (batch, sequence, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = self.rope(q, k, positions)
        return q, k, v

    def merge_heads(self, context: torch.Tensor) -> torch.Tensor:
        batch, _, sequence, _ = context.shape
        merged = context.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        return self.out_proj(merged)

    @staticmethod
    def reject_cache(cache: Any, use_cache: bool) -> None:
        if cache is not None or use_cache:
            raise NotImplementedError("KV caching is reserved for a later milestone")

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        cache: Any = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

