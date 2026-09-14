from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import AttentionBase

def manual_causal_attention(q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                            ) -> torch.Tensor:

        head_dim = q.shape[-1]
        sequence_length = q.shape[2]

        scores = q @ k.transpose(-2, -1)
        scores = scores * (head_dim**-0.5)
        
        causal_mask = torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=q.device
        ).triu(diagonal=1)

        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
        return probabilities @ v

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
        context = manual_causal_attention(q,k,v)
        return self.merge_heads(context)

