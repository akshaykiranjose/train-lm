from __future__ import annotations

import torch

from efficient_attention_lm.config import ModelConfig
from efficient_attention_lm.model import DecoderLM


def test_tiny_batch_can_overfit() -> None:
    torch.manual_seed(123)
    config = ModelConfig(
        vocab_size=16,
        context_length=8,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_heads=2,
    )
    model = DecoderLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)
    x = torch.randint(0, config.vocab_size, (2, config.context_length))
    y = torch.randint(0, config.vocab_size, (2, config.context_length))
    initial = float(model.loss(x, y).detach())
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()
    final = float(model.loss(x, y))
    assert final < initial * 0.2, (initial, final)

