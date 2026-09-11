"""A small, auditable decoder-only language model baseline."""

from .config import ModelConfig
from .model import DecoderLM

__all__ = ["DecoderLM", "ModelConfig"]

