from __future__ import annotations

import math
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from .checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from .config import model_config_from_dict, resolve_project_path, save_config
from .data import RandomWindowSampler
from .logging import CSVLogger, write_json
from .model import DecoderLM
from .tokenizer import sha256_file


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, config: dict[str, Any]) -> float:
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    max_lr = float(optimizer["max_lr"])
    min_lr = float(optimizer["min_lr"])
    warmup = int(scheduler["warmup_steps"])
    end = int(scheduler["cosine_end_step"])
    if warmup > 0 and step < warmup:
        return (step / warmup) * max_lr
    if step <= end and end > warmup:
        ratio = (step - warmup) / (end - warmup)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coefficient * (max_lr - min_lr)
    return min_lr


def make_optimizer(model: DecoderLM, config: dict[str, Any]) -> torch.optim.AdamW:
    values = config["optimizer"]
    if str(values["type"]).lower() != "adamw":
        raise ValueError("only AdamW is supported")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(values["max_lr"]),
        betas=tuple(float(value) for value in values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )


def autocast_context(device: torch.device, precision: str):
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'")
    if precision == "fp32":
        return nullcontext()
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured bf16 precision is unsupported by this CUDA device")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


@torch.no_grad()
def evaluate_model(
    model: DecoderLM,
    sampler: RandomWindowSampler,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    num_batches = int(evaluation["num_batches"])
    starts = sampler.fixed_starts(batch_size * num_batches, int(evaluation["seed"]))
    losses: list[float] = []
    was_training = model.training
    model.eval()
    for index in range(num_batches):
        batch_starts = starts[index * batch_size : (index + 1) * batch_size]
        x, y = sampler.sample(batch_size, device=device, starts=batch_starts)
        with autocast_context(device, config["training"]["precision"]):
            logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        losses.append(float(loss))
    model.train(was_training)
    return sum(losses) / len(losses)

def installed_package_versions() -> dict[str, str]:
    packages = {
        dist.metadata["Name"] : dist.version
        for dist in importlib_metadata.distributions() if dist.metadata["Name"]
    }
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))

def nvidia_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None

    
def environment_metadata(
                        config: dict[str, Any], 
                        root: Path, 
                        model: DecoderLM,
                        run_started_at_utc: str,) -> dict[str, Any]:
    def optional_hash(path_value: str) -> str | None:
        path = resolve_project_path(root, path_value)
        return sha256_file(path) if path.exists() else None

    commit = None
    git_branch = None
    git_dirty = None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        git_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        git_dirty = bool(status.strip())

    except (OSError, subprocess.CalledProcessError):
        commit = None

    gpu_name = None
    gpu_vram = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_vram = properties.total_memory
    data = config["data"]
    return {
        "git_commit": commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "random_seed": int(config["seed"]),
        "run_started_at_utc": run_started_at_utc,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_driver_version": nvidia_driver_version(),
        "gpu_name": gpu_name,
        "gpu_vram_bytes": gpu_vram,
        "parameter_count": model.num_parameters,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "installed_packages": installed_package_versions(),
        "precision": config["training"]["precision"],
        "tokenizer_sha256": optional_hash(data["tokenizer"]),
        "dataset_file_sha256": {
            "train": optional_hash(data["train_file"]),
            "valid": optional_hash(data["valid_file"]),
        },
    }


def _memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_peak_memory_allocated_mb": 0.0,
            "gpu_peak_memory_reserved_mb": 0.0,
        }
    divisor = 1024**2
    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / divisor,
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / divisor,
        "gpu_peak_memory_allocated_mb": (torch.cuda.max_memory_allocated() / divisor),
        "gpu_peak_memory_reserved_mb": (torch.cuda.max_memory_reserved() / divisor),
    }


def train(
    config: dict[str, Any],
    *,
    root: Path,
    run_dir: Path,
    resume: Path | None = None,
    device: torch.device | None = None,
) -> Path:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    (run_dir / "samples.txt").touch(exist_ok=True)

    run_started_at_utc = datetime.now(timezone.utc).isoformat()

    set_seed(int(config["seed"]))
    model = DecoderLM(model_config_from_dict(config)).to(device)

    metadata =  environment_metadata(config, root, model, run_started_at_utc)
    metadata["status"] = "running"
    metadata["run_completed_at_utc"] = None

    write_json(run_dir / "metadata.json", metadata)

    optimizer = make_optimizer(model, config)
    data = config["data"]
    train_sampler = RandomWindowSampler(
        resolve_project_path(root, data["train_tokens"]),
        int(data["context_length"]),
        int(config["seed"]),
    )
    valid_sampler = RandomWindowSampler(
        resolve_project_path(root, data["valid_tokens"]),
        int(data["context_length"]),
        int(config["evaluation"]["seed"]),
    )
    step = 0
    tokens_seen = 0
    best_validation_loss = math.inf
    if resume is not None: # resume is the path 
        payload = load_checkpoint(resume, model=model, optimizer=optimizer, device=device)
        step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])
        best_value = payload.get("best_validation_loss")
        best_validation_loss = math.inf if best_value is None else float(best_value)
        restore_rng_state(payload["rng_state"])
        sampler_state = payload["rng_state"].get("sampler")
        if sampler_state is not None:
            train_sampler.load_state_dict(sampler_state)

    training = config["training"]
    micro_batch = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    context_length = int(data["context_length"])
    tokens_per_step = int(training["effective_batch_size"]) * context_length
    total_steps = int(training["optimizer_steps"])
    evaluation_interval = int(config["evaluation"]["interval_steps"])
    checkpoint_interval = int(config["checkpointing"]["interval_steps"])
    precision = str(training["precision"])
    logger = CSVLogger(run_dir / "metrics.csv")
    start_time = time.perf_counter()

    model.train()

    # logging validation at step zero

    if step == 0:
        initial_validation_loss = evaluate_model(
            model,
            valid_sampler,
            config,
            device,
        )

        logger.log(
            {
                "optimizer_step": 0,
                "tokens_seen": 0,
                "train_loss": None,
                "validation_loss": initial_validation_loss,
                "validation_perplexity": math.exp(initial_validation_loss),
                "learning_rate": cosine_warmup_lr(0, config),
                "gradient_norm": None,
                "wallclock_seconds": time.perf_counter() - start_time,
                "step_time_ms": None,
                "tokens_per_second": None,
                **_memory_metrics(device),
            }
        )
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(total=total_steps,
                    initial=step,
                    desc="Training",
                    unit="step",
                    dynamic_ncols=True)
    
    while step < total_steps:
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(accumulation):
            x, y = train_sampler.sample(micro_batch, device=device)
            with autocast_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            accumulated_loss += float(loss.detach()) / accumulation
        learning_rate = cosine_warmup_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(config["gradient_clipping"]["max_l2_norm"]),
                        )
        
        optimizer.step()
        step += 1
        tokens_seen += tokens_per_step
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_start

        validation_loss = None
        if step % evaluation_interval == 0 or step == total_steps:
            validation_loss = evaluate_model(model, valid_sampler, config, device)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    checkpoints / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    tokens_seen=tokens_seen,
                    config=config,
                    sampler_state=train_sampler.state_dict(),
                    best_validation_loss=best_validation_loss,
                )

        logger.log(
            {
                "optimizer_step": step,
                "tokens_seen": tokens_seen,
                "train_loss": accumulated_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": (
                    math.exp(validation_loss) if validation_loss is not None else None
                ),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "wallclock_seconds": time.perf_counter() - start_time,
                "step_time_ms": step_seconds * 1000,
                "tokens_per_second": tokens_per_step / step_seconds,
                **_memory_metrics(device),
            }
        )

        postfix = {
            "loss": f"{accumulated_loss:.4f}",
            "lr": f"{learning_rate:.2e}",
            "tok/s": f"{tokens_per_step / step_seconds:.0f}",
        }
        if validation_loss is not None:
            postfix["val_loss"] = f"{validation_loss:.4f}"

        progress.set_postfix(postfix)
        progress.update(1)


        if step % checkpoint_interval == 0 or step == total_steps:
            save_checkpoint(
                checkpoints / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
                sampler_state=train_sampler.state_dict(),
                best_validation_loss=best_validation_loss,
            )
    progress.close()

    #updating the metadata if/when training completes
    metadata["status"] = "completed"
    metadata["run_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "metadata.json", metadata)

    return checkpoints / "latest.pt"
