from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END_OF_TEXT = "<|endoftext|>"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def train_byte_level_bpe(
    train_file: str | Path,
    output_file: str | Path,
    *,
    vocab_size: int = 10_000,
    special_tokens: Iterable[str] = (END_OF_TEXT,),
) -> dict:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(train_file)], trainer)
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"trained vocabulary has {tokenizer.get_vocab_size()} entries, expected {vocab_size}"
        )
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(target))
    metadata = {
        "type": "byte_level_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": list(special_tokens),
        "training_file": str(Path(train_file)),
        "training_file_sha256": sha256_file(train_file),
        "tokenizer_sha256": sha256_file(target),
    }
    _write_json(target.parent / "metadata.json", metadata)
    return metadata


def encode_text_file(
    tokenizer_file: str | Path,
    input_file: str | Path,
    output_file: str | Path,
) -> int:
    """Encode line-by-line; ByteLevel treats newlines as boundaries, avoiding huge RAM use."""
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    if tokenizer.token_to_id(END_OF_TEXT) is None:
        raise ValueError(f"tokenizer does not contain required special token {END_OF_TEXT}")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    token_count = 0
    with Path(input_file).open("r", encoding="utf-8", newline="") as source, temporary.open(
        "wb"
    ) as destination:
        while lines := [line for _, line in zip(range(1024), source)]:
            for encoding in tokenizer.encode_batch(lines, add_special_tokens=False):
                ids = encoding.ids
                if ids:
                    if max(ids) > np.iinfo(np.uint16).max:
                        raise ValueError("token id exceeds uint16 range")
                    array = np.asarray(ids, dtype=np.uint16)
                    destination.write(array.tobytes())
                    token_count += len(ids)
    temporary.replace(target)
    return token_count


def write_encoded_metadata(
    output_dir: str | Path,
    *,
    tokenizer_file: str | Path,
    train_file: str | Path,
    valid_file: str | Path,
    train_tokens: str | Path,
    valid_tokens: str | Path,
    train_count: int,
    valid_count: int,
) -> dict:
    metadata = {
        "dtype": "uint16",
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "source_sha256": {
            "train": sha256_file(train_file),
            "valid": sha256_file(valid_file),
        },
        "encoded_sha256": {
            "train": sha256_file(train_tokens),
            "valid": sha256_file(valid_tokens),
        },
        "token_counts": {"train": train_count, "valid": valid_count},
    }
    _write_json(Path(output_dir) / "metadata.json", metadata)
    return metadata


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
