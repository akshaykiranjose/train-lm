from __future__ import annotations

import argparse

from efficient_attention_lm.config import load_config, project_root, resolve_project_path
from efficient_attention_lm.tokenizer import train_byte_level_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen TinyStories byte-level BPE")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root(args.config)
    data = config["data"]
    metadata = train_byte_level_bpe(
        resolve_project_path(root, data["train_file"]),
        resolve_project_path(root, data["tokenizer"]),
        vocab_size=int(data["vocab_size"]),
    )
    print(f"Saved tokenizer ({metadata['tokenizer_sha256']})")


if __name__ == "__main__":
    main()

