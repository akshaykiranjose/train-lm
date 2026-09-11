from __future__ import annotations

from dataclasses import replace

import torch

from efficient_attention_lm.attention.manual_softmax import ManualCausalSelfAttention
from efficient_attention_lm.attention.sdpa import SDPACausalSelfAttention
from efficient_attention_lm.config import ModelConfig


def test_attention_shapes(tiny_config: ModelConfig) -> None:
    attention = ManualCausalSelfAttention(tiny_config)
    for batch, sequence in ((1, 1), (2, 9), (3, 16)):
        x = torch.randn(batch, sequence, tiny_config.d_model)
        assert attention(x).shape == x.shape


def test_sdpa_matches_manual_attention(tiny_config: ModelConfig) -> None:
    torch.manual_seed(7)
    manual = ManualCausalSelfAttention(tiny_config).eval()
    sdpa = SDPACausalSelfAttention(replace(tiny_config, attention_type="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, 11, tiny_config.d_model)
    torch.testing.assert_close(manual(x), sdpa(x), rtol=1e-5, atol=1e-6)


def test_cache_interface_is_reserved(tiny_config: ModelConfig) -> None:
    attention = ManualCausalSelfAttention(tiny_config)
    x = torch.randn(1, 4, tiny_config.d_model)
    try:
        attention(x, use_cache=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("use_cache=True should be rejected until caching is implemented")

