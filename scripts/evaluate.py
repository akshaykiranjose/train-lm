from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from efficient_attention_lm.checkpoint import load_checkpoint
from efficient_attention_lm.config import load_config, model_config_from_dict
from efficient_attention_lm.data import RandomWindowSampler
from efficient_attention_lm.model import DecoderLM
from efficient_attention_lm.training import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved run on fixed windows")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    root = run_dir.parent.parent
    config = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderLM(model_config_from_dict(config)).to(device)
    payload = load_checkpoint(
        run_dir / "checkpoints" / f"{args.checkpoint}.pt", model=model, device=device
    )
    valid_path = Path(config["data"]["valid_tokens"])
    if not valid_path.is_absolute():
        valid_path = root / valid_path
    sampler = RandomWindowSampler(
        valid_path,
        int(config["data"]["context_length"]),
        int(config["evaluation"]["seed"]),
    )
    loss = evaluate_model(model, sampler, config, device)
    result = {
        "checkpoint": args.checkpoint,
        "step": int(payload["step"]),
        "tokens_seen": int(payload["tokens_seen"]),
        "validation_loss": loss,
        "validation_perplexity": math.exp(loss),
    }
    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

