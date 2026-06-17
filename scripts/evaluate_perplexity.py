"""Evalue un checkpoint GPT sur un corpus de validation tenu a l'ecart.

Donne deux mesures comparables entre modeles :
- perplexite (exp de la loss moyenne) : plus bas = mieux ;
- accuracy de prediction du token suivant : plus haut = mieux.

    python scripts/evaluate_perplexity.py --val_corpus data/clean-v3-val.txt
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model import (  # noqa: E402
    LazyBlockDataset,
    build_token_cache,
    load_tokenizer,
    token_accuracy,
)
from backend.paths import find_latest_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perplexite et accuracy de validation.")
    parser.add_argument("--val_corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints-v3"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer-v3/tokenizer.json"))
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint or find_latest_checkpoint(args.checkpoints_dir)
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to(device)
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer, model.config.n_positions)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            "Le tokenizer ne correspond pas au modèle "
            f"({len(tokenizer)} tokens vs vocab {model.config.vocab_size}). "
            "Passe le tokenizer avec lequel ce checkpoint a été entraîné."
        )

    block_size = min(args.block_size, model.config.n_positions)
    cache_path = build_token_cache(args.val_corpus, tokenizer, "continuous")
    dataset = LazyBlockDataset(cache_path, block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_accuracy = 0.0
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_accuracy += token_accuracy(outputs.logits, batch["input_ids"])
            batches += 1

    mean_loss = total_loss / max(batches, 1)
    print(f"Checkpoint   : {checkpoint.name}")
    print(f"Paramètres   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Blocs val    : {len(dataset)}")
    print(f"Loss moyenne : {mean_loss:.4f}")
    print(f"Perplexité   : {math.exp(mean_loss):.2f}")
    print(f"Accuracy     : {total_accuracy / max(batches, 1):.4f}")


if __name__ == "__main__":
    main()
