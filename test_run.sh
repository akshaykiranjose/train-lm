#! /bin/bash

uv run python scripts/train.py \
  --config configs/a1_tinystories.yaml \
  --run-name runpod_smoke \
  --max-steps 20