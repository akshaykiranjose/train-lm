from __future__ import annotations

import argparse
from pathlib import Path

from efficient_attention_lm.config import load_config, project_root, with_runtime_limits
from efficient_attention_lm.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the A1-aligned TinyStories LM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--max-steps", type=int)
    limit.add_argument("--max-tokens", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = with_runtime_limits(
        load_config(args.config), max_steps=args.max_steps, max_tokens=args.max_tokens
    )
    root = project_root(args.config)
    run_dir = root / "runs" / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"run directory is non-empty: {run_dir}; use --resume")
    checkpoint = train(config, root=root, run_dir=run_dir, resume=args.resume)
    print(f"Training complete: {checkpoint}")


if __name__ == "__main__":
    main()

