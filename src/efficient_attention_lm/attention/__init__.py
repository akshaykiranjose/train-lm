from __future__ import annotations

from typing import Type

from .base import AttentionBase
from .manual_softmax import ManualCausalSelfAttention
from .sdpa import SDPACausalSelfAttention

ATTENTION_REGISTRY: dict[str, Type[AttentionBase]] = {
    "manual_softmax": ManualCausalSelfAttention,
    "sdpa": SDPACausalSelfAttention,
}


def build_attention(name: str, *args, **kwargs) -> AttentionBase:
    try:
        attention_class = ATTENTION_REGISTRY[name]
    except KeyError as error:
        options = ", ".join(sorted(ATTENTION_REGISTRY))
        raise ValueError(f"unknown attention type {name!r}; choose from {options}") from error
    return attention_class(*args, **kwargs)


__all__ = [
    "ATTENTION_REGISTRY",
    "AttentionBase",
    "ManualCausalSelfAttention",
    "SDPACausalSelfAttention",
    "build_attention",
]

