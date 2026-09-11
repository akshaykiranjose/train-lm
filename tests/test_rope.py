from __future__ import annotations

import torch

from efficient_attention_lm.rope import RotaryEmbedding


def test_rope_shape_position_zero_and_determinism() -> None:
    rope = RotaryEmbedding(8, theta=10_000.0, initial_length=4)
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn(2, 3, 7, 8)
    positions = torch.arange(7)
    q_rotated, k_rotated = rope(q, k, positions)
    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape
    torch.testing.assert_close(q_rotated[..., 0, :], q[..., 0, :])
    torch.testing.assert_close(k_rotated[..., 0, :], k[..., 0, :])
    q_again, k_again = rope(q, k, positions)
    torch.testing.assert_close(q_rotated, q_again)
    torch.testing.assert_close(k_rotated, k_again)


def test_rope_extends_beyond_training_context() -> None:
    rope = RotaryEmbedding(8, initial_length=2)
    q = torch.randn(1, 1, 32, 8)
    result, _ = rope(q, q)
    assert result.shape == q.shape
    assert rope.cos_cache.shape[0] >= 32

