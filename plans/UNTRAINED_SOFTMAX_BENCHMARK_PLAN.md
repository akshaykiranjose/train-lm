# Untrained Manual Softmax vs. SDPA Benchmark Plan

## Purpose

Complete the inference systems benchmarks from Sections 13–19 of
`PROJECT_PLAN_small_gpt_attention.md` for:

1. manual causal softmax attention, and
2. PyTorch causal SDPA.

None of the work in this plan requires a trained model. Random Q/K/V tensors are
sufficient for the attention-only benchmark, and a randomly initialized model is
sufficient for prefill, decode, memory, and profiler measurements. Dense attention
runtime and memory depend on tensor shapes, dtype, implementation, and hardware,
not on whether the weights have been trained.

This phase measures systems behavior only. It does not measure language-model
quality, validation loss, perplexity, or generation quality. Those comparisons will
be added later using trained checkpoints.

---

## Existing Repository State

Relevant code already present:

```text
src/efficient_attention_lm/
├── model.py
├── block.py
└── attention/
    ├── base.py
    ├── manual_softmax.py
    └── sdpa.py
```

The repository already has:

- interchangeable `manual_softmax` and `sdpa` attention modules,
- identical Q/K/V and output projections for the two implementations,
- an FP32 numerical-equivalence test between manual softmax and SDPA,
- RoPE caches that can grow beyond the training context length,
- the exact A1 model configuration in `configs/a1_tinystories.yaml`.

Still required:

- attention-only benchmark code,
- common CUDA timing and memory measurement code,
- full-model prefill benchmark code,
- KV caching for decode,
- decode benchmark code,
- CSV result output and plots,
- selected PyTorch Profiler runs after the timing results are stable.

---

## Fixed Benchmark Configuration

Use the implemented A1 model dimensions so the operator and full-model results
describe the same architecture:

```text
d_model: 512
num_heads: 16
head_dim: 32
num_layers: 4
vocab_size: 10000
batch_size: 1
dtype: bfloat16
```

Sweep context length:

```text
128
256
512
1024
2048
4096
8192
```

Run a length only where GPU memory permits. If a workload runs out of memory,
record the result as OOM and retain the shorter successful measurements.

For fair comparisons, create the manual-softmax model first and load the same
randomly initialized state dictionary into the SDPA model. No optimizer, dataset,
tokenizer, training loop, or checkpoint is involved.

---

## Benchmark 0 — Numerical Equivalence Check

This is a correctness gate before timing.

For several sequence lengths, pass identical inputs and identical random weights
through manual softmax and SDPA in evaluation mode. Verify that their outputs agree
within an appropriate tolerance.

The existing unit test already establishes this for a small configuration. Extend
coverage to the actual benchmark dimensions and at least a few benchmark sequence
lengths.

Do not mix correctness measurement into timed trials.

---

## Benchmark 1 — Attention Operator

### Workload

Create random tensors with shape:

```text
Q, K, V: [batch_size, num_heads, sequence_length, head_dim]
```

Benchmark only the causal attention computation:

```text
Manual: softmax((QK^T / sqrt(head_dim)) + causal_mask) V
SDPA:   scaled_dot_product_attention(Q, K, V, is_causal=True)
```

Q/K/V projection, RoPE, MLP, normalization, and the language-model head must not be
inside this timed region.

The current attention classes combine projection, RoPE, attention, and output
projection. Factor the actual attention calculation into a callable method or
helper so the operator can be benchmarked independently without changing its
mathematics.

### Measurements

For every implementation and sequence length, record:

- median attention latency in milliseconds,
- p10 latency,
- p90 latency,
- peak GPU memory allocated,
- peak GPU memory reserved.

### Purpose

This isolates the runtime and memory behavior of the two attention
implementations from the rest of the Transformer.

---

## Benchmark 2 — Full-Model Prefill

### Workload

Instantiate the complete four-layer model with random weights. Create random token
IDs with shape:

```text
[1, prompt_length]
```

Run the ordinary full-model forward pass in evaluation mode for every prompt
length. Use identical model weights and identical token IDs for manual softmax and
SDPA.

### Measurements

Record:

- total prefill latency,
- prompt tokens per second,
- peak GPU memory allocated,
- peak GPU memory reserved.

### Purpose

This shows how the attention implementation affects the complete model rather
than only the isolated operator.

The benchmark does not require meaningful text or trained weights because it does
not evaluate the logits for quality.

---

## Benchmark 3 — Autoregressive Decode

### Required implementation

Implement a conventional per-layer KV cache for both manual softmax and SDPA.

The minimum required behavior is:

1. process an initial random prompt and store its keys and values,
2. accept one new token,
3. append that token's key and value to the cache,
4. attend the new query to all cached keys and values,
5. return the new model output and updated cache.

Propagate the cache only through the files that need it:

```text
model.py
block.py
attention/base.py
attention/manual_softmax.py
attention/sdpa.py
```

Add a correctness test showing that cached token-by-token logits agree with logits
from recomputing the complete prefix.

### Workload

For every historical context length, first construct a KV cache using random token
IDs. Exclude that cache-building prefill from the decode timing. Then measure a
one-token cached forward step.

### Measurements

Record:

- milliseconds per generated token,
- generated tokens per second,
- KV-cache memory in bytes or MiB,
- total peak GPU memory allocated,
- total peak GPU memory reserved.

### Purpose

This measures how single-token inference behaves as the stored history grows.

---

## Timing Method Shared by All Benchmarks

CUDA execution is asynchronous, so use CUDA events rather than wall-clock timing
around an unsynchronized model call.

For each workload:

```text
warmup iterations: at least 20
measured iterations: 50–100
reported statistics: median, p10, p90
```

Required sequence:

1. allocate inputs before the timed region,
2. run warmup iterations,
3. synchronize CUDA,
4. record start and end CUDA events around only the intended workload,
5. synchronize CUDA,
6. save every measured latency,
7. compute the summary statistics.

Use `model.eval()` and `torch.inference_mode()` for full-model inference.

---

## Memory Measurement Shared by All Benchmarks

Before each isolated measurement:

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

For decode, separately calculate KV-cache memory from the number of stored tensor
elements and their element size.

---

## Minimal Repository Additions

Use the existing `benchmarks/` directory:

```text
benchmarks/
├── README.md
├── utils.py
├── attention_operator.py
├── prefill.py
├── decode.py
└── plot_results.py
```

Add one benchmark configuration:

```text
configs/benchmark_softmax.yaml
```

Write results to:

```text
benchmark_results/
├── attention_operator.csv
├── prefill.csv
├── decode.csv
└── figures/
```

Each CSV should identify at least:

```text
benchmark
attention_type
sequence_or_cache_length
batch_size
num_heads
head_dim
dtype
median_ms
p10_ms
p90_ms
tokens_per_second, where applicable
peak_memory_allocated_mb
peak_memory_reserved_mb
kv_cache_mb, for decode
status
```

Also record the GPU, PyTorch version, CUDA version, warmup count, and measured
iteration count with the benchmark results.

---

## Required Figures

Produce the plots already requested by the project plan:

1. attention-only latency vs. sequence length,
2. attention-only peak memory vs. sequence length,
3. full-model prefill latency vs. context length,
4. full-model prefill peak memory vs. context length,
5. decode milliseconds per token vs. historical context length,
6. decode KV-cache/peak memory vs. historical context length.

Plot manual softmax and SDPA together so their scaling can be compared directly.

---

## Profiling After the Benchmarks

Do not profile before the latency results are stable.

Use PyTorch Profiler on a small number of representative cases, such as one short
context and one long context for each implementation. Investigate only the
questions listed in the project plan:

- which CUDA kernels dominate runtime,
- whether many small kernels are launched,
- whether intermediate tensors are materialized,
- whether transposes or copies are expensive,
- whether SDPA uses optimized fused kernels,
- what explains the observed latency and memory difference.

Nsight Systems or Nsight Compute can remain optional unless PyTorch Profiler is
insufficient to explain a result.

---

## Execution Order

1. Add the shared CUDA timing and memory utilities.
2. Factor out the attention calculation needed by the operator benchmark.
3. Extend manual/SDPA equivalence tests to benchmark dimensions.
4. Implement and run the attention-operator sweep.
5. Implement and run the randomly initialized full-model prefill sweep.
6. Implement the KV cache and its cached-vs.-uncached correctness test.
7. Implement and run the decode sweep.
8. Save all measurements to CSV.
9. Generate the required plots.
10. Profile only representative results that need explanation.

---

## Definition of Done

This phase is complete when:

- manual softmax and SDPA pass numerical-equivalence checks,
- attention-only latency and memory have been measured across all feasible lengths,
- full-model prefill latency and memory have been measured across all feasible lengths,
- cached decode is correct,
- decode latency, throughput, KV-cache size, and peak memory have been measured,
- median, p10, and p90 are reported from repeated CUDA-event measurements,
- results are saved to CSV,
- the required comparison plots are generated,
- representative profiler traces explain the main observed implementation differences.

No model training or quality evaluation is part of this definition of done.

