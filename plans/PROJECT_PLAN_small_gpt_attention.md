# Small GPT Attention Benchmarking Project

## Project Goal

Build a small GPT-2-style causal language model from scratch, then use it as a controlled testbed for comparing standard softmax attention against alternative efficient/linear-attention mechanisms.

The project is not intended to be only an accuracy comparison. The main objective is to study the **quality–efficiency trade-off** of different attention mechanisms under realistic language-model inference workloads.

The central research question is:

> **For a fixed small causal language model and training budget, how do alternative attention mechanisms trade language-modeling quality against prefill latency, autoregressive decoding latency, and memory consumption as context length grows?**

The systems component is central. A theoretically more efficient attention mechanism is useful only if its advantages translate into measurable improvements on real hardware.

---

# 1. Project Scope

## Core comparison

Start with four attention variants:

1. **Naive causal softmax attention**
2. **PyTorch optimized causal softmax attention (SDPA)**
3. **Simple causal linear-attention baseline**
   - e.g. feature map  
     \[
     \phi(x)=\mathrm{ELU}(x)+1
     \]
4. **One or two paper-derived efficient-attention mechanisms**

Candidate papers from the existing reading list:

- FLatten Transformer — ICCV 2023
- Bridging the Divide: Reconsidering Softmax and Linear Attention — NeurIPS 2024
- Agent Attention — ECCV 2024
- Mobile Attention — ICML 2024
- PolaFormer — ICLR 2025
- Magnitude-Aware Linear Attention — ICCV 2025
- Breaking the Low-Rank Dilemma of Linear Attention — CVPR 2025
- SageAttention2 — ICML 2025
- MixA — ICCV 2025
- ELFATT — ACM Multimedia 2025
- SoLA-Vision — arXiv 2026
- ViT-AdaLA — arXiv 2026
- JetViT — arXiv 2026
- Finetuning Pretrained Transformers into RNNs — EMNLP 2021

The initial implementation should **not** attempt all of these. The goal is to build the experimental framework first, then select one or two mechanisms that can be meaningfully adapted to causal language modeling.

---

# 2. Important Constraint: Causal Language Modeling

A vision-attention formulation cannot automatically be inserted into a GPT decoder.

For every attention replacement, verify that token \(i\) cannot depend on any token \(j>i\).

Standard causal attention computes:

\[
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right)V
\]

where:

\[
M_{ij}=
\begin{cases}
0, & j\leq i \\
-\infty, & j>i
\end{cases}
\]

For kernelized linear attention, a causal implementation typically requires prefix accumulation:

\[
S_i=\sum_{j=1}^{i}\phi(k_j)v_j^\top
\]

\[
z_i=\sum_{j=1}^{i}\phi(k_j)
\]

\[
y_i=
\frac{\phi(q_i)S_i}
{\phi(q_i)z_i}
\]

This distinction is critical because a non-causal linear-attention implementation may accidentally use information from future tokens.

---

# 3. Dataset Plan

## Phase 0 — debugging

Use a tiny dataset such as Tiny Shakespeare only to verify that:

- the model runs,
- gradients flow,
- the model can overfit a tiny batch,
- checkpointing works,
- generation works.

This stage is not part of the final experimental comparison.

## Phase 1 — main experiments

Use **TinyStories**.

Reasons:

- small models learn meaningful language on it,
- faster iteration,
- easier debugging,
- practical for a local RTX 4060,
- suitable for models in the tens-of-millions-of-parameters range.

## Optional Phase 2

Repeat selected experiments on **WikiText-103**.

This should only happen after the complete pipeline works on TinyStories.

---

# 4. Model Configuration

## Debug model

Use this only while developing the training pipeline.

```yaml
n_layer: 4
n_head: 4
d_model: 256
d_ff: 1024
seq_len: 256
```

Goal:

> Verify correctness quickly.

## Main model

Initial target:

```yaml
n_layer: 8
n_head: 8
d_model: 512
head_dim: 64
d_ff: 2048
vocab_size: ~16000
train_seq_len: 512
max_seq_len: >=4096 if positional encoding permits
dropout: 0.0-0.1
dtype: bfloat16
optimizer: AdamW
```

Target parameter count:

> approximately 30–40M parameters.

The exact parameter count should be computed and logged.

Do not increase model size until the full benchmark pipeline is working.

---

# 5. GPT Architecture

The base model should remain deliberately conventional.

```text
tokens
  │
  ├── token embeddings
  │
  └── positional information
          │
          ▼
  Transformer block × N
          │
          ▼
       LayerNorm
          │
          ▼
 vocabulary projection
          │
          ▼
 next-token logits
```

Each transformer block:

```text
x
│
├─ LayerNorm
│     │
│     ▼
│  Attention
│     │
└──── + residual
      │
      ├─ LayerNorm
      │     │
      │     ▼
      │    MLP
      │     │
      └──── + residual
             │
             ▼
          output
```

The only component that should vary between experiments is the attention mechanism unless an experiment explicitly requires otherwise.

---

# 6. Codebase Design

Recommended repository structure:

```text
efficient-lm-attention/
│
├── README.md
├── PROJECT_PLAN.md
│
├── configs/
│   ├── debug.yaml
│   ├── gpt_35m.yaml
│   └── benchmark.yaml
│
├── src/
│   ├── model.py
│   ├── block.py
│   ├── mlp.py
│   ├── embeddings.py
│   │
│   └── attention/
│       ├── base.py
│       ├── softmax_naive.py
│       ├── softmax_sdpa.py
│       ├── linear_elu.py
│       ├── method_a.py
│       └── method_b.py
│
├── data/
│   ├── tokenizer.py
│   ├── tinystories.py
│   └── wikitext.py
│
├── benchmark/
│   ├── attention_operator.py
│   ├── prefill.py
│   ├── decode.py
│   ├── memory.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── generate.py
│
├── checkpoints/
│
└── results/
```

The transformer block should accept an interchangeable attention module.

Conceptually:

```python
class TransformerBlock(nn.Module):
    def __init__(self, config, attention_cls):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = attention_cls(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

The research code should make attention replacement isolated and explicit.

---

# 7. Baseline Attention Implementations

## 7.1 Naive causal softmax

Implement attention manually:

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right)
\]

\[
Y=AV
\]

The purpose is correctness and transparency.

This implementation may explicitly materialize the \(T\times T\) attention matrix.

---

## 7.2 Optimized softmax baseline

Implement a second version using:

```python
torch.nn.functional.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True,
)
```

This is the important practical softmax baseline.

The project should distinguish between:

```text
theoretical softmax attention
```

and

```text
a highly optimized GPU implementation of softmax attention
```

The latter may outperform a theoretically cheaper linear-attention formulation at short or moderate context lengths.

---

## 7.3 Simple causal linear attention

Before implementing a research paper, build a basic linear-attention baseline.

Example feature map:

\[
\phi(x)=\operatorname{ELU}(x)+1
\]

This provides:

- a correctness reference,
- a simple linear-time baseline,
- an intermediate implementation step,
- a way to distinguish paper-specific gains from generic linear attention.

---

# 8. Training Objective

Standard autoregressive next-token prediction.

Given tokens:

```text
[t0, t1, t2, ..., tn]
```

use:

```text
input  = [t0, t1, ..., t(n-1)]
target = [t1, t2, ..., tn]
```

Optimize cross entropy:

\[
\mathcal{L}
=
-\sum_t \log p(x_{t+1}\mid x_{\leq t})
\]

Main quality metrics:

- validation cross-entropy loss,
- validation perplexity.

Optional later metrics:

- zero-shot language-model tasks,
- generation quality inspection.

Do not make downstream benchmarking a requirement for the first version.

---

# 9. Training Protocol

All attention variants must be trained under as similar conditions as possible.

Hold constant:

- dataset,
- tokenizer,
- vocabulary,
- model width,
- number of layers,
- MLP dimensions,
- optimizer,
- learning-rate schedule,
- sequence length,
- token budget,
- initialization strategy where possible,
- random seeds,
- evaluation procedure.

Prefer comparing models by **tokens processed**, not wall-clock training time.

Example:

```text
Softmax-SDPA   : 200M tokens
ELU Linear     : 200M tokens
Method A       : 200M tokens
Method B       : 200M tokens
```

Training speed should then be reported separately.

---

# 10. Training Stages

## Stage A — tiny-batch overfit test

Train on a tiny fixed sample.

Expected behavior:

```text
loss
↓
↓
↓
approaches a very small value
```

If the model cannot memorize a tiny sample, do not continue.

Checklist:

- [ ] Forward pass works
- [ ] Backward pass works
- [ ] Loss decreases
- [ ] No NaNs
- [ ] Model can overfit a tiny batch

---

## Stage B — small TinyStories run

Run a short experiment to verify:

- [ ] training loss decreases
- [ ] validation loss decreases
- [ ] generated text becomes structured
- [ ] checkpoints save correctly
- [ ] checkpoints reload correctly
- [ ] evaluation reproduces saved metrics
- [ ] training can resume from checkpoint

---

## Stage C — controlled comparison

Only begin this after the pipeline is stable.

Train:

- [ ] Softmax-SDPA
- [ ] ELU causal linear attention
- [ ] Paper Method A
- [ ] Paper Method B

using a fixed token budget.

---

# 11. Training Metrics to Record

Every run should log:

```text
step
tokens_seen
train_loss
validation_loss
validation_perplexity
learning_rate
tokens_per_second
step_time
gpu_memory_allocated
gpu_memory_reserved
wall_clock_seconds
```

Also record run-level metadata:

```text
attention_type
parameter_count
dataset
tokenizer
vocab_size
training_sequence_length
batch_size
gradient_accumulation_steps
effective_batch_size
optimizer
learning_rate
weight_decay
warmup_steps
dtype
random_seed
GPU
PyTorch_version
CUDA_version
driver_version
git_commit
date
```

Prefer saving results locally in CSV/JSON even if an experiment tracker such as Weights & Biases is also used.

---

# 12. Results Directory

Example:

```text
results/
├── softmax_sdpa_seed42/
│   ├── config.yaml
│   ├── metadata.json
│   ├── training.csv
│   ├── validation.csv
│   ├── benchmarks.csv
│   └── checkpoint.pt
│
├── linear_elu_seed42/
│   └── ...
│
├── method_a_seed42/
│   └── ...
│
└── method_b_seed42/
    └── ...
```

Every experiment should be reproducible from its configuration file and git commit.

---

# 13. Benchmarking Philosophy

Training quality is only half of the project.

The key systems question is:

> **How do runtime and memory behave as sequence length grows?**

Benchmarks should therefore be separated into three categories.

---

# 14. Benchmark A — Attention Operator

Benchmark the attention operation by itself.

Input random Q/K/V tensors using fixed:

- batch size,
- number of heads,
- head dimension,
- dtype.

Sweep sequence length:

```text
128
256
512
1024
2048
4096
8192
```

where memory permits.

Measure:

- attention-only latency,
- peak allocated memory,
- peak reserved memory.

Purpose:

> Isolate the computational behavior of the attention algorithm from the rest of the transformer.

---

# 15. Benchmark B — Full-Model Prefill

Prefill means processing an existing prompt before generation begins.

Use:

```text
batch size = 1
```

Sweep prompt/context lengths:

```text
128
256
512
1024
2048
4096
8192
```

where supported.

Measure:

- total prefill latency,
- prompt tokens/second,
- peak allocated GPU memory,
- peak reserved GPU memory.

Primary graph:

```text
Prefill latency
      vs.
Context length
```

Secondary graph:

```text
Peak GPU memory
      vs.
Context length
```

---

# 16. Benchmark C — Autoregressive Decode

Measure token-by-token generation after a prompt has already been processed.

For softmax attention, maintain a KV cache.

Sweep historical context/cache size:

```text
128
256
512
1024
2048
4096
8192
```

Measure:

- milliseconds per generated token,
- generated tokens/second,
- cache/state memory,
- total peak GPU memory.

This benchmark is especially important because standard attention keeps historical K/V tensors.

For conventional attention:

\[
\text{KV cache memory} = O(Td)
\]

Some causal linear-attention formulations instead maintain a recurrent state independent of sequence length.

Example:

\[
S_t=S_{t-1}+\phi(k_t)v_t^\top
\]

\[
z_t=z_{t-1}+\phi(k_t)
\]

This makes decoding-state growth a central research question.

---

# 17. Benchmark Timing Method

CUDA operations are asynchronous.

Do not benchmark GPU execution using only:

```python
start = time.time()
model(x)
end = time.time()
```

Use CUDA events.

Example:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
model(x)
end.record()

torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end)
```

Before measurement, warm up the workload:

```python
for _ in range(20):
    model(x)

torch.cuda.synchronize()
```

Then run repeated trials.

Recommended:

```text
warmup iterations: 20+
measured iterations: 50–100+
```

Report at least:

- median,
- p10,
- p90.

Mean and standard deviation may also be recorded.

---

# 18. GPU Memory Measurement

Before each isolated benchmark:

```python
torch.cuda.reset_peak_memory_stats()
```

After the workload:

```python
torch.cuda.synchronize()
```

Record:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Both are useful because PyTorch's caching allocator can retain memory that is not currently occupied by live tensors.

---

# 19. Profiling

Use `torch.profiler` after coarse benchmarking reveals an interesting result.

Questions to investigate:

- Which CUDA kernels dominate runtime?
- Are there many small kernel launches?
- Are intermediate tensors being materialized?
- Are transposes/copies expensive?
- Is memory bandwidth limiting performance?
- Is the implementation poorly fused?
- Does SDPA exploit optimized fused kernels?
- At what context length does the theoretically efficient method become practically faster?

Possible tools later:

- PyTorch Profiler
- Nsight Systems
- Nsight Compute

Do not begin with low-level profiling before basic timing results are stable.

---

# 20. Expected Systems Phenomenon

A key hypothesis is that a theoretically better algorithm may not be faster at practical sequence lengths.

Possible result:

| Context | Softmax SDPA | Linear |
|---:|---:|---:|
| 128 | faster | slower |
| 512 | faster | slower |
| 1024 | similar | similar |
| 2048 | slower | faster |
| 4096 | much slower | faster |

This would imply a crossover context length:

\[
T^*
\]

such that:

```text
T < T*  -> optimized softmax is faster
T > T*  -> linear attention is faster
```

Finding and explaining this crossover point would be a strong systems result.

Potential explanations:

- kernel fusion,
- Tensor Core utilization,
- arithmetic intensity,
- memory bandwidth,
- kernel-launch overhead,
- temporary tensor allocation,
- poor implementation of the linear formulation,
- optimized SDPA/FlashAttention kernels.

---

# 21. Experimental Matrix

## Attention mechanisms

```text
Softmax-SDPA
ELU causal linear attention
Paper Method A
Paper Method B
```

## Model

```text
~30–40M parameters
```

## Dataset

```text
TinyStories
```

## Training budget

Choose one fixed budget, for example:

```text
100M–300M training tokens
```

Final budget should be selected after timing the baseline.

## Benchmark sequence lengths

```text
128
256
512
1024
2048
4096
8192
```

where technically feasible.

## Seeds

Initial development:

```text
1 seed
```

Final experiments if compute permits:

```text
3 seeds
```

---

# 22. Final Metrics

## Model quality

- validation loss
- validation perplexity
- loss vs training tokens
- perplexity vs training tokens

## Training efficiency

- training tokens/second
- step time
- training peak VRAM
- wall-clock time to fixed token budget

## Prefill

- latency
- tokens/second
- peak allocated VRAM
- peak reserved VRAM

## Autoregressive decode

- milliseconds/token
- tokens/second
- cache/state size
- peak VRAM

## Attention operator

- isolated operator latency
- isolated operator memory

---

# 23. Final Figures

Minimum target set of figures:

## Figure 1

```text
Validation perplexity
vs.
Training tokens
```

## Figure 2

```text
Prefill latency
vs.
Context length
```

## Figure 3

```text
Decode milliseconds/token
vs.
Context length
```

## Figure 4

```text
Peak GPU memory
vs.
Context length
```

## Figure 5

```text
Training throughput
vs.
Sequence length
```

Optional:

## Figure 6

```text
Attention-only latency
vs.
Sequence length
```

## Figure 7

```text
Quality vs latency Pareto frontier
```

---

# 24. Reproducibility Requirements

Each experiment should specify:

- [ ] configuration file
- [ ] random seed
- [ ] git commit
- [ ] package versions
- [ ] CUDA version
- [ ] GPU model
- [ ] tokenizer
- [ ] dataset version
- [ ] model parameter count
- [ ] precision
- [ ] training token budget
- [ ] benchmark warmup count
- [ ] benchmark repetition count

Avoid manually editing experimental code between runs without recording the change.

---

# 25. Things Not to Do Yet

Avoid expanding scope prematurely.

Do **not** initially:

- train a 100M+ model just because it is possible,
- implement five or ten attention papers,
- optimize custom CUDA kernels,
- benchmark multiple datasets,
- compare many model sizes,
- run downstream benchmarks,
- introduce speculative decoding,
- introduce quantization,
- compare different tokenizers,
- optimize for Raspberry Pi,
- combine multiple architectural changes simultaneously.

First establish a trustworthy experimental apparatus.

---

# 26. Milestones

## Milestone 0 — Project definition

- [x] Define research direction
- [x] Define quality + systems objectives
- [x] Decide to use a small causal GPT
- [x] Decide to benchmark context-length scaling
- [x] Separate prefill and decoding benchmarks
- [x] Identify attention replacement as the main experimental variable

---

## Milestone 1 — Baseline GPT

### Model

- [ ] Implement configuration system
- [ ] Implement token embeddings
- [ ] Implement positional encoding
- [ ] Implement LayerNorm
- [ ] Implement MLP
- [ ] Implement transformer block
- [ ] Implement GPT model
- [ ] Implement LM head
- [ ] Compute and log parameter count

### Attention

- [ ] Implement naive causal softmax
- [ ] Verify causal mask
- [ ] Implement SDPA causal softmax
- [ ] Numerically compare naive and SDPA outputs

### Training

- [ ] Implement next-token loss
- [ ] Implement optimizer
- [ ] Implement LR schedule
- [ ] Implement mixed precision
- [ ] Implement gradient clipping
- [ ] Implement checkpointing
- [ ] Implement resume-from-checkpoint

### Validation

- [ ] Overfit one tiny batch
- [ ] Train debug model
- [ ] Verify validation loss decreases
- [ ] Verify text generation

**Current next major goal: complete Milestone 1.**

---

## Milestone 2 — Data Pipeline

- [ ] Download/load TinyStories
- [ ] Choose tokenizer
- [ ] Train or load tokenizer
- [ ] tokenize dataset
- [ ] implement train/validation split
- [ ] implement packed fixed-length sequences
- [ ] verify next-token labels
- [ ] benchmark dataloader throughput

---

## Milestone 3 — Logging and Reproducibility

- [ ] Create run directory automatically
- [ ] save config
- [ ] save metadata
- [ ] save training CSV
- [ ] save validation CSV
- [ ] record GPU/CUDA/PyTorch environment
- [ ] record git commit
- [ ] record random seed
- [ ] save checkpoints

---

## Milestone 4 — Softmax Benchmark Harness

### Operator

- [ ] implement attention-only benchmark
- [ ] implement warmup
- [ ] implement CUDA-event timing
- [ ] implement repeated measurements
- [ ] record median/p10/p90

### Memory

- [ ] record max memory allocated
- [ ] record max memory reserved

### Full model

- [ ] implement prefill benchmark
- [ ] implement decode benchmark
- [ ] implement KV cache
- [ ] sweep context lengths

### Outputs

- [ ] latency vs context plot
- [ ] memory vs context plot
- [ ] decode latency vs context plot

---

## Milestone 5 — Simple Linear Attention

- [ ] implement ELU+1 feature map
- [ ] implement non-causal reference version
- [ ] derive causal prefix formulation
- [ ] implement causal training path
- [ ] implement recurrent decode state
- [ ] verify no future-token leakage
- [ ] compare against reference implementation
- [ ] train on TinyStories
- [ ] benchmark operator
- [ ] benchmark prefill
- [ ] benchmark decode
- [ ] benchmark memory

At this point, the full research pipeline should be operational.

---

## Milestone 6 — Select Paper Method A

Read candidate methods specifically from the perspective of causal LM adaptation.

Selection criteria:

- [ ] mathematically compatible with causal attention
- [ ] reasonable implementation complexity
- [ ] meaningful expected memory/runtime difference
- [ ] sufficiently different from ELU linear baseline
- [ ] can be trained within available compute
- [ ] does not require vision-specific assumptions that destroy the comparison

Candidate priority:

1. PolaFormer / related polarity-aware formulation
2. Bridging-the-Divide-style formulation
3. another clearly causalizable linear-attention method

Agent Attention / FLatten should be attempted only after understanding how their global/spatial mechanisms translate into strictly causal decoding.

---

## Milestone 7 — Paper Method A Experiments

- [ ] implement
- [ ] correctness tests
- [ ] causality tests
- [ ] train
- [ ] evaluate perplexity
- [ ] operator benchmark
- [ ] prefill benchmark
- [ ] decode benchmark
- [ ] memory benchmark
- [ ] profile bottlenecks

---

## Milestone 8 — Paper Method B

Repeat the same controlled protocol.

- [ ] select method
- [ ] implement
- [ ] correctness test
- [ ] train
- [ ] evaluate
- [ ] benchmark
- [ ] profile

---

## Milestone 9 — Final Controlled Runs

Freeze the code and experimental protocol.

Run:

- [ ] Softmax-SDPA
- [ ] ELU Linear
- [ ] Method A
- [ ] Method B

with identical:

- token budget,
- tokenizer,
- architecture,
- optimizer,
- schedule,
- dataset,
- evaluation procedure.

If compute allows:

- [ ] repeat using three seeds

---

## Milestone 10 — Analysis

Answer:

- [ ] Which attention gives best perplexity?
- [ ] Which trains fastest?
- [ ] Which uses least memory?
- [ ] Which has fastest short-context prefill?
- [ ] Which has fastest long-context prefill?
- [ ] Which has fastest decoding?
- [ ] How does cache/state memory grow?
- [ ] Is there a runtime crossover point?
- [ ] Does theoretical complexity predict observed latency?
- [ ] What kernels dominate runtime?
- [ ] Where is linear attention slower than expected?
- [ ] What implementation bottlenecks explain the results?

---

# 27. Final Research Story

A strong final narrative would be:

> We construct a controlled GPT-style language-model benchmark for comparing standard softmax attention with efficient attention mechanisms. Rather than evaluating only perplexity, we characterize training throughput, prefill latency, autoregressive decoding latency, and memory scaling across context lengths. We study the gap between asymptotic algorithmic complexity and realized GPU performance, identify crossover points where linear attention becomes beneficial, and analyze the implementation factors responsible for observed performance.

This is substantially stronger than:

> We replaced attention in GPT with another attention mechanism and compared perplexity.

---

# 28. Immediate Next Steps

Do these in order.

1. **Create repository skeleton.**
2. **Implement the small GPT model.**
3. **Implement naive causal softmax attention.**
4. **Make the model overfit a tiny batch.**
5. **Implement optimized SDPA attention.**
6. **Train a debug model on TinyStories.**
7. **Add structured logging and checkpointing.**
8. **Build the softmax benchmark harness.**
9. **Produce the first latency-vs-context and memory-vs-context plots.**
10. **Only then begin causal linear attention.**

The next concrete deliverable should therefore be:

> **A working ~30–40M GPT baseline plus a benchmark script that measures softmax prefill latency and memory against context length.**

---

# 29. Project Success Criteria

The project is successful if it produces:

- a clean modular GPT implementation,
- at least one correct causal linear-attention implementation,
- at least one paper-derived efficient-attention implementation,
- controlled language-model quality comparisons,
- rigorous prefill benchmarks,
- rigorous decode benchmarks,
- memory-scaling measurements,
- reproducible configurations,
- clear plots,
- and an explanation of when theoretical efficiency does or does not translate into practical speedup.

A paper submission is a possible later outcome, but the first goal is to build a technically strong, reproducible ML-systems project.
