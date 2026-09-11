from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block import TransformerBlock
from .config import ModelConfig
from .norm import RMSNorm


def initialize_a1_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        d_out, d_in = module.weight.shape
        sigma = (2.0 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(module.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.gain)


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(initialize_a1_weights)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("token sequence must not be empty")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.final_norm(x))

    def loss(self, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(token_ids)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
