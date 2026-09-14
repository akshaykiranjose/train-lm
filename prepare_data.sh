#!/bin/bash

uv run python scripts/download_data.py
uv run python scripts/train_tokenizer.py --config configs/a1_tinystories.yaml
uv run python scripts/encode_dataset.py --config configs/a1_tinystories.yaml