from __future__ import annotations

import torch

from efficient_attention_lm.config import ModelConfig
from efficient_attention_lm.model import DecoderLM


def test_future_tokens_do_not_change_past_logits(tiny_config: ModelConfig) -> None:
    torch.manual_seed(11)
    model = DecoderLM(tiny_config).eval()
    tokens_a = torch.randint(0, tiny_config.vocab_size, (2, 12))
    tokens_b = tokens_a.clone()
    position = 5
    tokens_b[:, position + 1 :] = torch.randint(
        0, tiny_config.vocab_size, tokens_b[:, position + 1 :].shape
    )
    with torch.no_grad():
        logits_a = model(tokens_a)
        logits_b = model(tokens_b)
    torch.testing.assert_close(
        logits_a[:, : position + 1], logits_b[:, : position + 1], rtol=0, atol=1e-6
    )

