from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1_344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = False
    bias: bool = False
    attention_type: str = "manual_softmax"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def validate(self) -> None:
        if min(
            self.vocab_size,
            self.context_length,
            self.d_model,
            self.d_ff,
            self.num_layers,
            self.num_heads,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.dropout != 0.0:
            raise ValueError("the A1 baseline requires dropout=0.0")
        if self.tie_embeddings:
            raise ValueError("the A1 baseline requires untied embeddings")
        if self.bias:
            raise ValueError("the A1 baseline requires bias-free linear layers")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def model_config_from_dict(config: dict[str, Any]) -> ModelConfig:
    data = config["data"]
    model = config["model"]
    result = ModelConfig(
        vocab_size=int(data["vocab_size"]),
        context_length=int(data["context_length"]),
        **model,
    )
    result.validate()
    return result


def validate_config(config: dict[str, Any]) -> None:
    required = {"data", "model", "training", "optimizer", "scheduler"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    training = config["training"]
    micro = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    effective = int(training["effective_batch_size"])
    if micro * accumulation != effective:
        raise ValueError(
            "micro_batch_size * gradient_accumulation_steps must equal "
            "effective_batch_size"
        )
    context = int(config["data"]["context_length"])
    steps = int(training["optimizer_steps"])
    total = int(training["total_tokens"])
    if effective * context * steps != total:
        raise ValueError(
            "effective_batch_size * context_length * optimizer_steps must "
            "equal total_tokens"
        )
    model_config_from_dict_unchecked(config).validate()
    if not 0 < int(config["data"]["vocab_size"]) <= 65_535:
        raise ValueError("uint16 encoding requires vocab_size <= 65535")


def model_config_from_dict_unchecked(config: dict[str, Any]) -> ModelConfig:
    data = config["data"]
    return ModelConfig(
        vocab_size=int(data["vocab_size"]),
        context_length=int(data["context_length"]),
        **config["model"],
    )


def with_runtime_limits(
    config: dict[str, Any],
    *,
    max_steps: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Return a resolved config whose token/step invariant remains exact."""
    result = deepcopy(config)
    tokens_per_step = (
        int(result["training"]["effective_batch_size"])
        * int(result["data"]["context_length"])
    )
    if max_steps is not None and max_tokens is not None:
        raise ValueError("specify only one of max_steps and max_tokens")
    if max_tokens is not None:
        if max_tokens < tokens_per_step:
            raise ValueError(f"max_tokens must be at least {tokens_per_step}")
        max_steps = max_tokens // tokens_per_step
        result.setdefault("runtime", {})["requested_max_tokens"] = int(max_tokens)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        result["training"]["optimizer_steps"] = int(max_steps)
        result["training"]["total_tokens"] = int(max_steps) * tokens_per_step
        result.setdefault("runtime", {})["max_steps"] = int(max_steps)
    validate_config(result)
    return result


def project_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"could not find pyproject.toml above {path}")


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
