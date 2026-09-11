from __future__ import annotations

import random

import numpy as np
import torch

from efficient_attention_lm.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from efficient_attention_lm.config import ModelConfig
from efficient_attention_lm.model import DecoderLM


def test_checkpoint_restores_training_and_rng(tmp_path, tiny_config: ModelConfig) -> None:
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    model = DecoderLM(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 8))
    model.loss(tokens, targets).backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=3,
        tokens_seen=384,
        config={"test": True},
        sampler_state={"cursor": 17},
        best_validation_loss=2.5,
    )
    expected_random = (random.random(), np.random.random(), torch.rand(3))
    next_tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    next_targets = torch.randint(0, tiny_config.vocab_size, (2, 8))
    optimizer.zero_grad(set_to_none=True)
    model.loss(next_tokens, next_targets).backward()
    optimizer.step()

    restored_model = DecoderLM(tiny_config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    payload = load_checkpoint(
        path, model=restored_model, optimizer=restored_optimizer, device="cpu"
    )
    restore_rng_state(payload["rng_state"])
    actual_random = (random.random(), np.random.random(), torch.rand(3))
    resumed_tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    resumed_targets = torch.randint(0, tiny_config.vocab_size, (2, 8))
    assert payload["step"] == 3
    assert payload["tokens_seen"] == 384
    assert payload["rng_state"]["sampler"] == {"cursor": 17}
    assert expected_random[0] == actual_random[0]
    assert expected_random[1] == actual_random[1]
    torch.testing.assert_close(expected_random[2], actual_random[2])
    torch.testing.assert_close(next_tokens, resumed_tokens)
    torch.testing.assert_close(next_targets, resumed_targets)
    restored_optimizer.zero_grad(set_to_none=True)
    restored_model.loss(resumed_tokens, resumed_targets).backward()
    restored_optimizer.step()
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(expected, actual)
