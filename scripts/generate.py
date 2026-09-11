from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from efficient_attention_lm.checkpoint import load_checkpoint
from efficient_attention_lm.config import load_config, model_config_from_dict
from efficient_attention_lm.model import DecoderLM
from efficient_attention_lm.tokenizer import END_OF_TEXT


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities > top_p
    sorted_probabilities[remove] = 0
    sorted_probabilities /= sorted_probabilities.sum()
    sampled = torch.multinomial(sorted_probabilities, 1)
    return int(sorted_indices[sampled])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a saved run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    args = parser.parse_args()
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    run_dir = args.run.resolve()
    root = run_dir.parent.parent
    config = load_config(run_dir / "config.yaml")
    tokenizer_path = Path(config["data"]["tokenizer"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = root / tokenizer_path
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderLM(model_config_from_dict(config)).to(device)
    load_checkpoint(
        run_dir / "checkpoints" / f"{args.checkpoint}.pt", model=model, device=device
    )
    model.eval()
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")
    end_id = tokenizer.token_to_id(END_OF_TEXT)
    if end_id is None:
        raise ValueError(f"tokenizer is missing {END_OF_TEXT}")
    context_length = int(config["data"]["context_length"])
    for _ in range(args.max_new_tokens):
        context = torch.tensor([token_ids[-context_length:]], device=device)
        next_id = sample_top_p(model(context)[0, -1], args.temperature, args.top_p)
        token_ids.append(next_id)
        if next_id == end_id:
            break
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(text)
    with (run_dir / "samples.txt").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


if __name__ == "__main__":
    main()
