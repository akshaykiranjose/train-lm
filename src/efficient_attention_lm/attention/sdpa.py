from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import AttentionBase


class SDPACausalSelfAttention(AttentionBase):
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
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        return self.merge_heads(context)

