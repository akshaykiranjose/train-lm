from __future__ import annotations

import torch
from torch import nn

from .attention import build_attention
from .config import ModelConfig
from .ffn import SwiGLU
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = build_attention(config.attention_type, config)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), positions=positions)
        return x + self.ffn(self.norm2(x))

