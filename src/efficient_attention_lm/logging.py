from __future__ import annotations

import csv
import json
from pathlib import Path


METRIC_FIELDS = [
    "optimizer_step",
    "tokens_seen",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "gradient_norm",
    "wallclock_seconds",
    "step_time_ms",
    "tokens_per_second",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "gpu_peak_memory_allocated_mb",
    "gpu_peak_memory_reserved_mb",
]


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._has_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, values: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            if not self._has_header:
                writer.writeheader()
                self._has_header = True
            writer.writerow({field: values.get(field, "") for field in METRIC_FIELDS})


def write_json(path: str | Path, values: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")

