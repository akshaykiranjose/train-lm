from __future__ import annotations

import argparse

from efficient_attention_lm.config import load_config, project_root, resolve_project_path
from efficient_attention_lm.tokenizer import encode_text_file, write_encoded_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode TinyStories as uint16 streams")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root(args.config)
    data = config["data"]
    tokenizer = resolve_project_path(root, data["tokenizer"])
    train_source = resolve_project_path(root, data["train_file"])
    valid_source = resolve_project_path(root, data["valid_file"])
    train_target = resolve_project_path(root, data["train_tokens"])
    valid_target = resolve_project_path(root, data["valid_tokens"])
    print("Encoding training data...")
    train_count = encode_text_file(tokenizer, train_source, train_target)
    print("Encoding validation data...")
    valid_count = encode_text_file(tokenizer, valid_source, valid_target)
    write_encoded_metadata(
        train_target.parent,
        tokenizer_file=tokenizer,
        train_file=train_source,
        valid_file=valid_source,
        train_tokens=train_target,
        valid_tokens=valid_target,
        train_count=train_count,
        valid_count=valid_count,
    )
    print(f"Encoded {train_count:,} train and {valid_count:,} validation tokens")


if __name__ == "__main__":
    main()

