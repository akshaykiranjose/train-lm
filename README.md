# Efficient Attention LM

This repository provides a clean decoder-only language-model baseline for controlled efficient-attention experiments. Its data, tokenizer specification, model dimensions, and training token budget follow the Stanford CS336 Spring 2025 Assignment 1 TinyStories experiment. Standard libraries replace coursework exercises that are not research-relevant here: BPE internals, AdamW, and cross entropy.

This is an independent research implementation, not an official Stanford implementation or a coursework submission.

## Baseline specification

- Exact `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt` sources used by CS336.
- Frozen 10,000-token byte-level BPE trained only on the training file, with `<|endoftext|>` registered as a special token.
- Decoder-only, pre-norm Transformer: context 256, width 512, SwiGLU width 1,344, 4 layers, 16 heads, and 32 dimensions per head.
- Full-head RoPE at theta 10,000, RMSNorm epsilon `1e-5`, no dropout or linear biases, and untied input/output embeddings.
- Exactly 22,696,448 trainable parameters.
- 5,000 optimizer steps at effective batch size 256: 327,680,000 training tokens.
- Manual causal softmax attention is the reference; PyTorch SDPA is an interchangeable optimized baseline.

## Setup and data

Install [uv](https://docs.astral.sh/uv/), then run from this directory:

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/train_tokenizer.py --config configs/a1_tinystories.yaml
uv run python scripts/encode_dataset.py --config configs/a1_tinystories.yaml
uv run pytest -q
```

The downloader uses the exact URLs named by the Spring 2025 assignment. Tokenizer and encoded-data metadata include SHA256 hashes. Encoded files are reusable `uint16` streams, and training samples are independent shifted windows read through `numpy.memmap`.

## Training

```bash
# Stage A: code and logging smoke test
uv run python scripts/train.py \
  --config configs/a1_tinystories.yaml \
  --run-name smoke \
  --max-steps 20

# Stage B: short learning run (rounded down to a whole effective batch)
uv run python scripts/train.py \
  --config configs/a1_tinystories.yaml \
  --run-name short_baseline \
  --max-tokens 10000000

# Stage C: full baseline
uv run python scripts/train.py \
  --config configs/a1_tinystories.yaml \
  --run-name a1_manual_softmax
```

CLI limits are written into the resolved run config while preserving the experiment invariant. If micro-batch 16 exceeds available VRAM, lower it and increase gradient accumulation by the same factor so their product remains 256.

Resume without restarting the schedule:

```bash
uv run python scripts/train.py \
  --config runs/a1_manual_softmax/config.yaml \
  --run-name a1_manual_softmax \
  --resume runs/a1_manual_softmax/checkpoints/latest.pt
```

Each run records resolved configuration, environment and data hashes, per-step CSV metrics, deterministic validation, latest/best checkpoints, and a samples file under `runs/<name>/`.

## Evaluation and generation

```bash
uv run python scripts/evaluate.py --run runs/a1_manual_softmax
uv run python scripts/generate.py \
  --run runs/a1_manual_softmax \
  --prompt "Once upon a time" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

## Results

Full training has not yet been run in this fresh scaffold; results remain explicit rather than estimated.

| Attention | Tokens | Val Loss | PPL | Train tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Manual softmax | 327.68M | TBD | TBD | TBD | TBD |
| PyTorch SDPA | 327.68M | TBD | TBD | TBD | TBD |

## Extension points and reproducibility

Attention implementations inherit from `AttentionBase` and are selected by `model.attention_type`. Future linear, POLA, or agent attention variants only need a new module and registry entry; the Transformer and training stack remain unchanged. RoPE caches grow beyond the training context for later long-sequence operator benchmarks, while cache arguments reserve room for KV-cache or recurrent-state work.

All architecture comparisons should reuse the frozen tokenizer, encoded streams, validation seed/windows, effective batch, optimizer configuration, precision, and total token budget. Do not add paper-derived attention variants until the manual baseline passes the included correctness tests and Stage B learning run.

Authoritative reference: [Stanford CS336 Assignment 1, Spring 2025](https://github.com/stanford-cs336/assignment1-basics/tree/spring2025).
