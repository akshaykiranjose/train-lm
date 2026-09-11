from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import AttentionBase


class ManualCausalSelfAttention(AttentionBase):
    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        cache: Any = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        self.reject_cache(cache, use_cache)
        q, k, v = self.project_qkv(x, positions)
        scores = q @ k.transpose(-2, -1)
        scores = scores * (self.head_dim**-0.5)
        sequence = x.shape[1]
        causal_mask = torch.ones(
            sequence, sequence, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
        return self.merge_heads(probabilities @ v)

