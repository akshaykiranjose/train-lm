from __future__ import annotations

import pytest

from efficient_attention_lm.config import ModelConfig


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        context_length=16,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        attention_type="manual_softmax",
    )

