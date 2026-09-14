#imports required
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from efficient_attention_lm.attention.manual_softmax import manual_causal_attention
from efficient_attention_lm.attention.sdpa import sdpa_causal_attention

from efficient_attention_lm.config import (
    load_config,
    model_config_from_dict
)

PROJECT_ROOT = Path(__file__).resolve().parents[1] #one-level higher parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "a1_tinystories.yaml"

SEQUENCE_LENGHTS = [128, 256, 512]#, 1024, 2048, 4096]
BATCH_SIZE = 1
WARMUP_ITERS = 2#0
MEASURED_ITERS = 5#0

#write the benchmark_operator function:

@torch.inference_mode()
def benchmark_operator(
    operator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
) -> dict[str, float]:

    #warmup iterations:
    for _ in range(WARMUP_ITERS):
        operator(q,k,v)


    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    elapsed_times = []

    for _ in range(MEASURED_ITERS):

        start.record()
        output = operator(q,k,v)
        end.record()

        end.synchronize()
        elapsed_times.append(start.elapsed_time(end))

    torch.cuda.synchronize()

    times = torch.tensor(elapsed_times, dtype=torch.float64)

    return {
        "median_ms": float(times.median()),
        "p10_ms": float(torch.quantile(times, 0.10)),
        "p90_ms": float(torch.quantile(times, 0.90)),
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "peak_reserved_mb": (
            torch.cuda.max_memory_reserved() / 1024**2
        ),
    }

#sweep the lengths and time everything.


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmarking required CUDA GPU")


    #create a layer with the current config with the attention layer as required
    config = load_config(CONFIG_PATH)
    model_config = model_config_from_dict(config)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    operators = {
        "manual_softmax": manual_causal_attention,
        "sdpa": sdpa_causal_attention,
    }

    for seq_len in SEQUENCE_LENGHTS:

        shape = (BATCH_SIZE, model_config.num_heads, seq_len, model_config.head_dim)

        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)

        print(f"Sequence length: {seq_len}")

        for name, operator in operators.items():

            try:

                result = benchmark_operator(operator, q,k,v)

                print(
                    f"{name:16s} "
                    f"median={result['median_ms']:.3f} ms  "
                    f"p10={result['p10_ms']:.3f} ms  "
                    f"p90={result['p90_ms']:.3f} ms  "
                    f"allocated={result['peak_allocated_mb']:.1f} MiB  "
                    f"reserved={result['peak_reserved_mb']:.1f} MiB"
                )

            except torch.OutOfMemoryError:
                print(f"{name:16s} OOM")
                torch.cuda.empty_cache()

        del q,k,v
        torch.cuda.empty_cache()

if __name__ == "__main__":

    main()