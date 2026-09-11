from __future__ import annotations

import torch
import torch.nn.functional as F

from efficient_attention_lm.config import ModelConfig
from efficient_attention_lm.ffn import SwiGLU
from efficient_attention_lm.model import DecoderLM
from efficient_attention_lm.norm import RMSNorm


def test_exact_a1_parameter_count() -> None:
    model = DecoderLM(ModelConfig())
    assert model.num_parameters == 22_696_448
    assert model.token_embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()


def test_model_shapes(tiny_config: ModelConfig) -> None:
    model = DecoderLM(tiny_config)
    for batch, sequence in ((1, 1), (2, 7), (3, 16)):
        token_ids = torch.randint(0, tiny_config.vocab_size, (batch, sequence))
        assert model(token_ids).shape == (batch, sequence, tiny_config.vocab_size)


def test_rmsnorm_matches_fp32_definition() -> None:
    layer = RMSNorm(8, eps=1e-5)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    actual = layer(x)
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5)
        * layer.gain.float()
    ).to(x.dtype)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == x.dtype


def test_swiglu_matches_direct_expression() -> None:
    layer = SwiGLU(8, 16)
    x = torch.randn(2, 3, 8)
    expected = F.linear(
        F.silu(F.linear(x, layer.w1.weight)) * F.linear(x, layer.w3.weight),
        layer.w2.weight,
    )
    torch.testing.assert_close(layer(x), expected)


def test_initialization_is_explicit_and_bounded(tiny_config: ModelConfig) -> None:
    model = DecoderLM(tiny_config)
    assert float(model.token_embedding.weight.detach().abs().max()) <= 3.0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            d_out, d_in = module.weight.shape
            sigma = (2 / (d_in + d_out)) ** 0.5
            assert float(module.weight.detach().abs().max()) <= 3 * sigma + 1e-6

