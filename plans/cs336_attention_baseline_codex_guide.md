# Coding Agent Guide: CS336 Assignment-1-Aligned Baseline for Efficient-Attention LM Experiments

## 0. Purpose

Build a **clean, reproducible decoder-only language-model baseline** that matches the **Stanford CS336 Spring 2025 Assignment 1 TinyStories model and training setup** wherever Assignment 1 specifies the setup, while deliberately using standard libraries for components that are not the focus of this research project.

This repository is **not** an attempt to complete CS336 Assignment 1 as coursework. The goal is to get a trustworthy baseline running quickly so that subsequent work can focus on:

1. implementing alternative causal attention mechanisms,
2. training controlled attention variants,
3. benchmarking attention/prefill/decode latency,
4. measuring memory and state/KV-cache scaling,
5. profiling GPU behavior.

The baseline must therefore be:
- faithful to the CS336 A1 model/data setup,
- modular at the attention boundary,
- deterministic/reproducible where practical,
- instrumented enough for later systems experiments,
- simple enough to audit.

---

# 1. Authoritative reference

Use the **Spring 2025 archive** of Stanford CS336 Assignment 1 as the reference, not the current `main` branch if it has changed.

Official Spring 2025 repository:

https://github.com/stanford-cs336/assignment1-basics/tree/spring2025

Official Spring 2025 handout:

https://github.com/stanford-cs336/assignment1-basics/blob/spring2025/cs336_spring2025_assignment1_basics.pdf

The implementation should reproduce the **TinyStories baseline** described in Section 7.2 of the handout.

Do **not** copy random student solutions as authoritative references.

---

# 2. What must match Assignment 1

The following are experiment-defining choices and must be preserved.

## 2.1 Dataset

Use the exact TinyStories files used by CS336 Spring 2025:

- `TinyStoriesV2-GPT4-train.txt`
- `TinyStoriesV2-GPT4-valid.txt`

Official download sources:

```text
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Do not silently substitute another TinyStories release or a Hugging Face split with different preprocessing.

The OpenWebText portion of Assignment 1 is **not required for the first baseline milestone**. Keep the repository extensible enough to add it later, but the required experiment in this guide is TinyStories.

---

## 2.2 Tokenizer specification

Assignment 1 trains a byte-level BPE tokenizer on TinyStories with:

```yaml
tokenizer:
  type: byte_level_bpe
  vocab_size: 10000
  special_tokens:
    - "<|endoftext|>"
```

For this project, **do not implement BPE training from scratch**.

Use a maintained tokenizer library such as Hugging Face `tokenizers` to train a byte-level BPE tokenizer on the **exact TinyStories training text**.

Requirements:

- vocabulary size: **10,000**
- byte-level BPE
- `<|endoftext|>` registered as a special token
- train tokenizer on the TinyStories **training file only**
- encode both train and validation files using the same frozen tokenizer
- preserve `<|endoftext|>` document delimiters
- save the tokenizer artifact to disk
- save a SHA256 hash of the tokenizer artifact in run metadata
- use the **same tokenizer artifact for every attention architecture**

The exact integer IDs/merge tie-breaking do not have to match a student's from-scratch BPE implementation. What matters for this research project is that the tokenizer specification and training corpus match A1 and that every attention variant uses the exact same frozen tokenizer.

Serialize encoded token streams as `uint16`, since the vocabulary is < 65,536.

Recommended output:

```text
data/
├── raw/
│   ├── TinyStoriesV2-GPT4-train.txt
│   └── TinyStoriesV2-GPT4-valid.txt
├── tokenizer/
│   ├── tokenizer.json
│   └── metadata.json
└── encoded/
    ├── train.bin
    ├── valid.bin
    └── metadata.json
```

---

# 3. Exact baseline model architecture

Implement the decoder-only Transformer used in CS336 A1.

## 3.1 Fixed model configuration

```yaml
model:
  vocab_size: 10000
  context_length: 256

  d_model: 512
  d_ff: 1344

  num_layers: 4
  num_heads: 16
  head_dim: 32

  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5

  dropout: 0.0
  tie_embeddings: false
  bias: false
```

Important:

- `head_dim = d_model / num_heads = 32`
- no learned absolute positional embeddings
- use RoPE
- no linear biases
- input embedding and LM output head are **not weight-tied**
- do not silently convert this into GPT-2's LayerNorm/GELU/learned-position architecture
- do not use `GPT2LMHeadModel`

Expected parameter count, assuming untied embeddings and no biases:

```text
input token embedding:             5,120,000
4 Transformer blocks:            12,455,936
final RMSNorm:                           512
LM output projection:             5,120,000
------------------------------------------------
total:                           22,696,448
```

The handout describes this as roughly **17M non-input-embedding parameters**.

Add a unit test that verifies the total trainable parameter count is exactly `22_696_448`.

---

# 4. Transformer block semantics

Use a **pre-norm** Transformer block.

For each block:

```python
x = x + attention(rmsnorm_1(x))
x = x + swiglu(rmsnorm_2(x))
```

After all blocks:

```python
x = final_rmsnorm(x)
logits = lm_head(x)
```

The LM head is an independent, bias-free linear projection from `d_model` to `vocab_size`.

---

# 5. RMSNorm

Use the A1 RMSNorm behavior:

```text
RMSNorm(x) = gain * x / sqrt(mean(x^2) + eps)
```

Requirements:

- learnable gain initialized to ones
- `eps = 1e-5`
- compute normalization statistics in FP32 even when model activations use BF16
- cast the normalized result back to the input dtype

Using `torch.nn.RMSNorm` is acceptable **only if its behavior is explicitly configured and verified** against this definition. A tiny local implementation is also fine.

---

# 6. SwiGLU FFN

Use the A1 SwiGLU structure with three bias-free matrices:

```python
hidden = F.silu(W1(x)) * W3(x)
out = W2(hidden)
```

Shapes:

```text
W1: d_model -> d_ff
W3: d_model -> d_ff
W2: d_ff    -> d_model
```

with:

```text
d_model = 512
d_ff    = 1344
```

Do not replace SwiGLU with a standard GELU MLP.

---

# 7. Baseline causal self-attention

The initial A1-faithful baseline should use ordinary causal multi-head self-attention.

Input:

```text
x: [B, T, D]
```

Projection:

```text
q, k, v: [B, H, T, Dh]
```

with:

```text
H  = 16
Dh = 32
D  = 512
```

Apply RoPE to **Q and K**.

Then compute:

```text
scores = Q @ K^T / sqrt(Dh)
scores[future_positions] = -inf
probs = softmax(scores, dim=-1)
context = probs @ V
```

Merge heads and apply the output projection.

## 7.1 Important baseline choice

Implement the default `manual` attention path explicitly using PyTorch tensor operations. This is the closest conceptual match to Assignment 1 and will serve as the architecture-correctness baseline.

Also leave room for a later second baseline:

```text
sdpa
```

using:

```python
torch.nn.functional.scaled_dot_product_attention(...)
```

Do **not** make SDPA the only implementation because later experiments need to distinguish:

1. mathematical/architecture changes,
2. kernel/implementation changes.

---

# 8. Attention must be a replaceable module

This is the most important deviation from a one-off homework implementation.

The rest of the Transformer must not know which attention architecture it is using.

Use a registry or factory such as:

```python
ATTENTION_REGISTRY = {
    "manual_softmax": ManualCausalSelfAttention,
    "sdpa": SDPACausalSelfAttention,
}
```

and instantiate blocks from configuration:

```yaml
model:
  attention_type: manual_softmax
```

Design the interface now so future modules can be added without rewriting the Transformer.

Recommended conceptual API:

```python
class AttentionBase(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        cache=None,
        use_cache: bool = False,
    ):
        ...
```

For the initial training baseline, caching may remain unused.

Future implementations should be addable as:

```text
attention/
├── base.py
├── manual_softmax.py
├── sdpa.py
├── linear_elu.py
├── pola.py
├── agent.py
└── ...
```

Do not implement the paper-derived attention modules yet.

---

# 9. RoPE

Implement RoPE in a small, auditable module.

Requirements:

```yaml
rope:
  theta: 10000.0
  dimensions: full_head_dim
```

Apply RoPE to Q and K after reshaping into heads and before the QK dot product.

Precompute/cache sin/cos tables up to the configured context length where convenient.

The baseline context length is 256, but write RoPE so the cache can later be extended to longer benchmark sequences without rewriting the model.

---

# 10. Parameter initialization

Match the A1 initialization rules.

## Linear weights

For a linear weight with input width `d_in` and output width `d_out`:

```text
sigma = sqrt(2 / (d_in + d_out))
```

Initialize from:

```text
Normal(mean=0, std=sigma)
```

truncated to:

```text
[-3*sigma, +3*sigma]
```

Use `torch.nn.init.trunc_normal_`.

No biases.

## Token embeddings

Initialize embeddings from:

```text
Normal(mean=0, std=1)
```

truncated to:

```text
[-3, +3]
```

## RMSNorm

Initialize gain to `1`.

Do not use PyTorch defaults silently. Put initialization in one explicit function and test basic statistics/shapes.

---

# 11. Data loading semantics

After tokenization, treat each split as one long sequence of integer token IDs.

For every batch element:

```python
start = random valid index

x = tokens[start : start + context_length]
y = tokens[start + 1 : start + context_length + 1]
```

Return:

```text
x: [B, 256]
y: [B, 256]
```

Use memory mapping rather than loading duplicated training windows into RAM.

Recommended:

```python
np.memmap(..., dtype=np.uint16, mode="r")
```

The batch sampler should sample random independent start positions.

For reproducibility, allow the sampler RNG seed to be configured.

---

# 12. Training token budget

The CS336 A1 TinyStories baseline specifies approximately:

```text
327,680,000 total training tokens
```

The invariant is:

```text
effective_batch_size * context_length * optimizer_steps
≈ 327,680,000
```

For this project, choose the following **fixed reproducible run configuration**:

```yaml
training:
  context_length: 256
  effective_batch_size: 256
  optimizer_steps: 5000
  total_tokens: 327680000
```

because:

```text
256 sequences
× 256 tokens
× 5000 optimizer updates
= 327,680,000 tokens
```

The RTX 4060 does **not** need to hold 256 sequences simultaneously.

Use gradient accumulation.

Example:

```yaml
training:
  micro_batch_size: 16
  gradient_accumulation_steps: 16
  effective_batch_size: 256
```

Allow `micro_batch_size` to be lowered if required by VRAM while preserving:

```text
micro_batch_size * gradient_accumulation_steps = 256
```

All future attention variants must use the same:

- effective batch size,
- context length,
- total token budget,
- optimizer hyperparameters,
- tokenizer,
- dataset,
- model width/depth/FFN.

---

# 13. Optimizer and learning-rate schedule

Important distinction:

**CS336 A1 does not prescribe one exact final learning rate, warmup, AdamW beta choice, or weight decay for the TinyStories experiment. It asks students to tune these.**

Therefore, the architecture/data/token budget above are A1-fixed, while the optimizer values below are **project-fixed defaults** chosen to make all future attention comparisons reproducible.

Use PyTorch's implementation rather than writing AdamW from scratch.

Recommended project baseline:

```yaml
optimizer:
  type: AdamW
  max_lr: 1.0e-3
  min_lr: 1.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1

scheduler:
  type: cosine_with_linear_warmup
  warmup_steps: 500
  cosine_end_step: 5000

gradient_clipping:
  max_l2_norm: 1.0
```

Schedule:

```text
if step < warmup_steps:
    lr = step / warmup_steps * max_lr

elif step <= cosine_end_step:
    cosine decay from max_lr to min_lr

else:
    lr = min_lr
```

Once the first healthy baseline is established, **freeze these optimizer settings** for attention-comparison experiments.

If the baseline is unstable, adjust the baseline configuration once, record the change, and then freeze the revised configuration before beginning architecture comparisons.

---

# 14. Precision

Assignment 1 does not define one mandatory training precision for the TinyStories run.

For the RTX 4060, support:

```text
bf16 autocast
fp32
```

Default project run:

```yaml
precision: bf16
```

if the local GPU/PyTorch combination supports BF16 correctly.

Keep:

- RMSNorm statistics in FP32,
- loss calculation numerically stable,
- optimizer state behavior standard PyTorch.

All compared attention architectures must use the same precision.

Do not enable quantization.

---

# 15. Loss

Use ordinary next-token cross entropy.

It is acceptable and preferred to use:

```python
torch.nn.functional.cross_entropy
```

instead of reimplementing cross entropy.

Input logits:

```text
[B, T, vocab_size]
```

Targets:

```text
[B, T]
```

Flatten consistently for loss calculation.

Validation perplexity:

```text
perplexity = exp(validation_loss)
```

Report validation loss as the primary quantity; perplexity is derived.

---

# 16. Checkpointing

Each checkpoint must contain at minimum:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": optimizer_step,
    "tokens_seen": tokens_seen,
    "config": full_config,
    "rng_state": ...,
}
```

Save:

```text
latest.pt
best.pt
```

Resume must restore:

- model,
- optimizer,
- training step,
- LR schedule position,
- token count.

Do not silently restart the LR schedule after resuming.

---

# 17. Logging and experiment metadata

Logging is part of the research infrastructure, not optional polish.

Write every run to a unique directory:

```text
runs/
└── <run_name>/
    ├── config.yaml
    ├── metrics.csv
    ├── metadata.json
    ├── checkpoints/
    │   ├── latest.pt
    │   └── best.pt
    └── samples.txt
```

Record at least:

```text
optimizer_step
tokens_seen
train_loss
validation_loss
validation_perplexity
learning_rate
wallclock_seconds
step_time_ms
tokens_per_second
gpu_memory_allocated_mb
gpu_memory_reserved_mb
gpu_peak_memory_allocated_mb
```

Metadata must include:

```text
git commit
random seed
Python version
PyTorch version
CUDA runtime version
GPU name
GPU VRAM
precision
tokenizer SHA256
dataset file SHA256 hashes
```

Weights & Biases may be supported as an optional logger, but **CSV/JSON logging must work without W&B**.

---

# 18. Validation protocol

Validation should be deterministic enough to compare experiments.

Use one of these approaches:

1. precompute a fixed set of validation batch start indices, or
2. use a dedicated validation RNG with a fixed seed.

Do not let different model runs evaluate on different randomly sampled validation windows.

Recommended:

```yaml
evaluation:
  interval_steps: 100
  num_batches: 50
  batch_size: 32
  seed: 12345
```

The exact evaluation frequency is a project choice; the same protocol must be used for every attention variant.

---

# 19. Required correctness tests

Do not launch the full 327.68M-token run until these pass.

## 19.1 Shape tests

Verify:

```text
model([B,T]) -> [B,T,V]
attention([B,T,D]) -> [B,T,D]
```

for several B/T combinations up to context 256.

## 19.2 Parameter-count test

Assert:

```python
sum(p.numel() for p in model.parameters()) == 22_696_448
```

for the exact A1 configuration.

## 19.3 Causality test

This test is mandatory.

Take a token sequence and compute model outputs.

Then alter tokens strictly after position `t`.

Assert that logits for positions `<= t` remain unchanged to numerical tolerance.

Example idea:

```python
tokens_a = [...]
tokens_b = tokens_a.clone()
tokens_b[:, t+1:] = random_other_tokens

logits_a = model(tokens_a)
logits_b = model(tokens_b)

assert_close(logits_a[:, :t+1], logits_b[:, :t+1])
```

This catches future-token leakage.

## 19.4 RoPE test

Verify:

- output shape unchanged,
- position 0 rotation behaves correctly,
- same position/QK treatment is deterministic.

## 19.5 SwiGLU test

Verify the implementation against a direct expression using the same weights.

## 19.6 Tiny-batch overfit test

Before the real run:

- take a tiny fixed batch,
- train repeatedly on it,
- verify loss falls dramatically.

If the model cannot overfit a tiny batch, do not run the real experiment.

## 19.7 Checkpoint-resume test

Train a few steps, checkpoint, resume, and verify the resumed run continues correctly.

---

# 20. Required smoke-training stages

Use three levels.

## Stage A — code sanity

```text
model: exact A1 model
dataset: TinyStories
steps: 20-50
```

Goal:

- no NaNs,
- no OOM,
- correct logging/checkpointing,
- loss finite.

## Stage B — short learning run

```text
tokens: ~5-10M
```

Goal:

- training loss clearly decreases,
- validation loss decreases,
- generated text begins to show TinyStories structure.

## Stage C — full A1-aligned run

```text
total tokens: 327,680,000
context: 256
effective batch size: 256
optimizer steps: 5000
```

This is the baseline checkpoint used for later comparisons.

Do not start implementing paper-derived attention methods until Stage B passes.

---

# 21. Text generation

Provide a simple generation script for qualitative sanity checks.

Requirements:

- tokenize a prompt with the frozen tokenizer,
- autoregressively generate tokens,
- stop on `<|endoftext|>` or max token count,
- support temperature,
- support top-p,
- decode to text.

This is not a major research component, so keep it small.

Example:

```bash
python -m scripts.generate \
    --checkpoint runs/a1_softmax/checkpoints/best.pt \
    --prompt "Once upon a time" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-p 0.95
```

---

# 22. Repository structure

Use approximately:

```text
efficient-attention-lm/
├── pyproject.toml
├── README.md
├── configs/
│   └── a1_tinystories.yaml
│
├── data/
│   ├── raw/
│   ├── tokenizer/
│   └── encoded/
│
├── src/
│   └── efficient_attention_lm/
│       ├── config.py
│       ├── model.py
│       ├── block.py
│       ├── rope.py
│       ├── norm.py
│       ├── ffn.py
│       │
│       ├── attention/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── manual_softmax.py
│       │   └── sdpa.py
│       │
│       ├── data.py
│       ├── tokenizer.py
│       ├── training.py
│       ├── checkpoint.py
│       └── logging.py
│
├── scripts/
│   ├── download_data.py
│   ├── train_tokenizer.py
│   ├── encode_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── benchmarks/
│   ├── README.md
│   └── placeholder.md
│
├── tests/
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_causality.py
│   ├── test_data.py
│   └── test_checkpoint.py
│
└── runs/
```

The `benchmarks/` directory may remain mostly empty in this first milestone. Do not spend time building the full Assignment 2 benchmarking suite before the baseline trains correctly.

---

# 23. Libraries: what to use vs. what to implement

## Use existing libraries for

Use standard, well-tested implementations for:

```text
BPE training/tokenization       -> Hugging Face tokenizers
AdamW                           -> torch.optim.AdamW
cross entropy                   -> torch.nn.functional.cross_entropy
dataset downloading             -> huggingface_hub / standard HTTP tooling
configuration                   -> dataclasses + YAML, or equivalent
CSV/JSON logging                -> standard libraries / pandas optional
```

PyTorch's standard `nn.Linear` and `nn.Embedding` are acceptable for this research code, provided:

- linear layers use `bias=False`,
- A1 initialization is applied explicitly,
- embedding initialization is applied explicitly.

## Implement/audit locally

Keep these local and easy to inspect:

```text
Transformer block
RMSNorm behavior
SwiGLU composition
RoPE
causal attention
attention interface/registry
training loop
checkpointing wrapper
validation protocol
```

The point is not to prove that we can rewrite PyTorch primitives. The point is to own the pieces that determine the architecture and experimental methodology.

---

# 24. Do not use Hugging Face Transformer model classes

Do not instantiate:

```text
GPT2LMHeadModel
AutoModelForCausalLM
LlamaForCausalLM
```

as the baseline.

They introduce architectural/default differences and make later attention replacement less transparent.

Hugging Face/tokenizer libraries are fine for **tokenization/data utilities**.

The model itself should remain a small local PyTorch implementation.

---

# 25. Do not prematurely optimize

For the first training baseline:

- do not write Triton kernels,
- do not use custom CUDA,
- do not add FlashAttention packages,
- do not add quantization,
- do not add distributed training,
- do not add FSDP,
- do not add activation checkpointing unless VRAM forces it,
- do not use `torch.compile` by default.

First establish a correct numerical baseline.

Optimization comes after correctness.

---

# 26. Research-oriented extension points to prepare now

The code should make these future experiments possible without refactoring the training stack.

## 26.1 Alternative attention

Future config:

```yaml
model:
  attention_type: linear_elu
```

or:

```yaml
model:
  attention_type: pola
```

Only the attention module should change.

## 26.2 Longer benchmark context

Although A1 training uses context 256, later inference/operator benchmarks will test:

```text
128
256
512
1024
2048
4096
possibly 8192
```

Do not hard-code RoPE or masks to 256 in a way that prevents longer benchmark inputs.

Training remains at context 256 for the A1 baseline.

## 26.3 KV cache / recurrent state

The interface should leave room for:

```text
softmax attention -> KV cache
linear attention  -> recurrent state
```

Do not implement the full cache system yet unless it is trivial to do cleanly.

## 26.4 Attention-only benchmarking

Attention modules should be independently callable so they can later be benchmarked without the rest of the Transformer.

---

# 27. Initial SDPA variant

After the manual baseline passes all tests, add an **optional** SDPA attention implementation using:

```python
torch.nn.functional.scaled_dot_product_attention
```

It must use the exact same:

- Q/K/V projections,
- RoPE,
- output projection,
- model parameters/shapes.

Add an equivalence test in evaluation mode:

```text
manual_softmax output ≈ SDPA output
```

within appropriate numerical tolerance.

This gives two useful baselines later:

```text
manual_softmax = architecture-faithful/reference implementation
sdpa           = optimized PyTorch softmax baseline
```

For systems experiments, SDPA will generally be the more meaningful real-world softmax comparator.

---

# 28. Config file to create

Create:

```text
configs/a1_tinystories.yaml
```

with approximately:

```yaml
seed: 42

data:
  name: TinyStoriesV2-GPT4
  train_file: data/raw/TinyStoriesV2-GPT4-train.txt
  valid_file: data/raw/TinyStoriesV2-GPT4-valid.txt
  train_tokens: data/encoded/train.bin
  valid_tokens: data/encoded/valid.bin
  tokenizer: data/tokenizer/tokenizer.json
  vocab_size: 10000
  context_length: 256

model:
  d_model: 512
  d_ff: 1344
  num_layers: 4
  num_heads: 16
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5
  dropout: 0.0
  tie_embeddings: false
  bias: false
  attention_type: manual_softmax

training:
  total_tokens: 327680000
  effective_batch_size: 256
  micro_batch_size: 16
  gradient_accumulation_steps: 16
  optimizer_steps: 5000
  precision: bf16

optimizer:
  type: AdamW
  max_lr: 1.0e-3
  min_lr: 1.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1

scheduler:
  type: cosine_with_linear_warmup
  warmup_steps: 500
  cosine_end_step: 5000

gradient_clipping:
  max_l2_norm: 1.0

evaluation:
  interval_steps: 100
  num_batches: 50
  batch_size: 32
  seed: 12345

checkpointing:
  interval_steps: 500
```

The coding agent should validate:

```text
micro_batch_size * gradient_accumulation_steps
== effective_batch_size
```

and:

```text
effective_batch_size * context_length * optimizer_steps
== total_tokens
```

Do not silently change these experiment invariants.

If `micro_batch_size=16` OOMs, lower it and increase accumulation proportionally.

---

# 29. Command-line workflow

The finished baseline should support a workflow like:

```bash
# 1. Environment
uv sync

# 2. Download the exact CS336 TinyStories files
uv run python scripts/download_data.py

# 3. Train the 10k byte-level BPE tokenizer
uv run python scripts/train_tokenizer.py \
    --config configs/a1_tinystories.yaml

# 4. Encode train and validation
uv run python scripts/encode_dataset.py \
    --config configs/a1_tinystories.yaml

# 5. Run tests
uv run pytest -q

# 6. Tiny smoke run
uv run python scripts/train.py \
    --config configs/a1_tinystories.yaml \
    --run-name smoke \
    --max-steps 20

# 7. Short learning run
uv run python scripts/train.py \
    --config configs/a1_tinystories.yaml \
    --run-name short_baseline \
    --max-tokens 10000000

# 8. Full baseline
uv run python scripts/train.py \
    --config configs/a1_tinystories.yaml \
    --run-name a1_manual_softmax

# 9. Evaluate
uv run python scripts/evaluate.py \
    --run runs/a1_manual_softmax

# 10. Generate
uv run python scripts/generate.py \
    --run runs/a1_manual_softmax \
    --prompt "Once upon a time"
```

CLI overrides must be saved into the final resolved `config.yaml` inside the run directory.

---

# 30. README requirements

Create a concise README containing:

1. project purpose,
2. relation to Stanford CS336 Spring 2025 Assignment 1,
3. explicit statement that standard libraries replace non-research-relevant A1 implementation exercises,
4. exact dataset/model configuration,
5. setup commands,
6. training commands,
7. current baseline results,
8. future attention modules,
9. reproducibility notes.

Do not claim this repository is an official Stanford implementation.

---

# 31. Baseline results table

The README should eventually contain:

```markdown
| Attention | Tokens | Val Loss | PPL | Train tok/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Manual softmax | 327.68M | TBD | TBD | TBD | TBD |
| PyTorch SDPA | 327.68M | TBD | TBD | TBD | TBD |
```

Do not add alternative paper architectures until the manual baseline is validated.

---

# 32. Definition of done for this coding-agent task

The baseline milestone is complete only when all of the following are true:

- [ ] Exact CS336 TinyStories train/validation source files are downloaded.
- [ ] A 10k byte-level BPE tokenizer with `<|endoftext|>` is trained and frozen.
- [ ] Train/validation corpora are encoded to reusable `uint16` token streams.
- [ ] The model uses the exact A1 dimensions: 512 / 1344 / 4 layers / 16 heads / context 256.
- [ ] The model uses pre-norm RMSNorm.
- [ ] The model uses SwiGLU.
- [ ] The model uses RoPE with theta 10000.
- [ ] The model uses causal MHA.
- [ ] Linear layers have no bias.
- [ ] Input embedding and LM head are not tied.
- [ ] Parameter count is exactly 22,696,448.
- [ ] Causality unit test passes.
- [ ] Tiny-batch overfit test passes.
- [ ] Checkpoint-resume test passes.
- [ ] Smoke training runs without NaNs/OOM.
- [ ] Short TinyStories training clearly decreases train/validation loss.
- [ ] Full 327.68M-token configuration is runnable.
- [ ] Metrics are logged locally to CSV/JSON.
- [ ] Run metadata records software/hardware/dataset/tokenizer versions.
- [ ] Attention is replaceable through a clean registry/interface.
- [ ] Manual softmax attention remains available as the correctness reference.
- [ ] Optional PyTorch SDPA implementation can be added without changing the rest of the model.

---

# 33. Explicit non-goals for this milestone

Do **not** spend time on:

- implementing BPE from scratch,
- implementing AdamW from scratch,
- implementing cross entropy from scratch,
- reproducing every CS336 Assignment 1 exercise,
- OpenWebText leaderboard tuning,
- Assignment 1 architecture ablations,
- distributed training,
- Triton,
- FlashAttention2 implementation,
- custom CUDA kernels,
- alternative attention papers,
- full inference benchmarking,
- publication-quality plots.

Those are not required to get the baseline ready.

---

# 34. What comes immediately after this milestone

Once the baseline passes the definition of done:

## Phase 2A — optimized softmax baseline

Add:

```text
SDPA causal attention
```

and verify numerical agreement with manual softmax.

## Phase 2B — benchmarking harness

Implement:

```text
attention-only latency
prefill latency
decode latency
peak memory
KV-cache size
```

with proper CUDA events and warmups.

## Phase 2C — first alternative attention

Implement a simple causal linear-attention control such as:

```text
phi(x) = ELU(x) + 1
```

before attempting a paper-derived mechanism.

Then begin the controlled attention-comparison project.

---

# 35. Guiding principle for the coding agent

When there is tension between "reimplement everything from Assignment 1" and "move quickly toward the attention research", choose the latter **unless the choice changes the model, dataset, tokenization specification, training token budget, or experimental comparability**.

Spend implementation effort on the parts that will later determine the scientific result:

```text
attention semantics
causality
RoPE/QKV handling
training reproducibility
validation
inference state/cache
profiling
benchmarking
```

Use standard libraries for commodity functionality.

Do not introduce architectural improvements merely because they are more modern or convenient. The first baseline exists to be a stable, controlled reference.

