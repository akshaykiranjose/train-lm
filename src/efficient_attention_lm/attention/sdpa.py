from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import AttentionBase

def sdpa_causal_attention(
                            q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                        ) -> torch.Tensor:

        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

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
        context = sdpa_causal_attention(q,k,v)
        return self.merge_heads(context)

