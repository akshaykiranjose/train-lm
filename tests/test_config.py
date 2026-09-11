from __future__ import annotations

import pytest

from efficient_attention_lm.config import load_config, with_runtime_limits


def test_baseline_config_invariants() -> None:
    config = load_config("configs/a1_tinystories.yaml")
    assert config["training"]["effective_batch_size"] == 256
    assert config["training"]["total_tokens"] == 327_680_000
    smoke = with_runtime_limits(config, max_steps=20)
    assert smoke["training"]["optimizer_steps"] == 20
    assert smoke["training"]["total_tokens"] == 20 * 256 * 256
    short = with_runtime_limits(config, max_tokens=10_000_000)
    assert short["runtime"]["requested_max_tokens"] == 10_000_000
    assert short["training"]["total_tokens"] <= 10_000_000


def test_invalid_batch_invariant_is_rejected() -> None:
    config = load_config("configs/a1_tinystories.yaml")
    config["training"]["gradient_accumulation_steps"] = 2
    with pytest.raises(ValueError, match="micro_batch_size"):
        with_runtime_limits(config)
