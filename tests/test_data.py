from __future__ import annotations

import numpy as np
import torch

from efficient_attention_lm.data import RandomWindowSampler


def test_memmap_sampler_returns_shifted_windows(tmp_path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.uint16).tofile(path)
    sampler = RandomWindowSampler(path, context_length=8, seed=4)
    starts = np.array([0, 10, 91])
    x, y = sampler.sample(3, starts=starts)
    assert x.shape == y.shape == (3, 8)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    assert x.dtype == y.dtype == torch.int64
    assert int(x[-1, -1]) == 98
    assert int(y[-1, -1]) == 99


def test_sampler_rng_can_resume(tmp_path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.uint16).tofile(path)
    first = RandomWindowSampler(path, context_length=8, seed=9)
    first.sample(2)
    state = first.state_dict()
    expected = first.sample(2)
    resumed = RandomWindowSampler(path, context_length=8, seed=0)
    resumed.load_state_dict(state)
    actual = resumed.sample(2)
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])

