"""Pré-entraîne un petit BERT from scratch par masquage de tokens."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.paths import BERT_MLM_DIR, BERT_TOKENIZER_DIR, DATA_DIR  # noqa: E402


class MLMDataset(Dataset):
    def __init__(
        self,
        corpus_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
    ) -> None:
        lines = [
            line.strip()
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.examples = [
            tokenizer(
                line,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
            for line in lines
        ]
        if not self.examples:
            raise ValueError("Le corpus MLM est vide.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Exactitude sur les seuls tokens masqués (labels != -100)."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    return (predictions[mask] == labels[mask]).float().mean().item()


def save_checkpoint(
    model: BertForMaskedLM,
    tokenizer: PreTrainedTokenizerFast,
    output_root: Path,
    epoch: int,
    global_step: int,
    average_loss: float,
) -> Path:
    base_name = f"checkpoint-epoch-{epoch}-step-{global_step}"
    checkpoint = output_root / base_name
    suffix = 2
    while checkpoint.exists():
        checkpoint = output_root / f"{base_name}-run-{suffix}"
        suffix += 1
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    (checkpoint / "training_info.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "global_step": global_step,
                "average_loss": average_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint.name}, indent=2),
        encoding="utf-8",
    )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pré-entraîne le mini-BERT en MLM.")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "clean.txt")
    parser.add_argument("--tokenizer_dir", type=Path, default=BERT_TOKENIZER_DIR)
    parser.add_argument("--output_dir", type=Path, default=BERT_MLM_DIR)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Fréquence (en updates) d'écriture des métriques pour la visualisation.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("--hidden_size doit être divisible par --num_attention_heads.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    dataset = MLMDataset(args.corpus, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(256, args.max_length),
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    planned_steps = updates_per_epoch * args.epochs
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_steps * 0.05)),
        num_training_steps=max(total_steps, 1),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Appareil             : {device}")
    print(f"Phrases MLM          : {len(dataset)}")
    print(f"Paramètres mini-BERT : {parameter_count:,}")
    print(f"Étapes prévues       : {total_steps}")

    global_step = 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics_handle = (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        progress = tqdm(dataloader, desc=f"MLM époque {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_update = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(dataloader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    accuracy = masked_accuracy(outputs.logits, batch["labels"])
                    metrics_handle.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "loss": round(outputs.loss.item(), 4),
                                "accuracy": round(accuracy, 4),
                                "lr": scheduler.get_last_lr()[0],
                                "elapsed_s": round(time.time() - start_time, 1),
                            }
                        )
                        + "\n"
                    )
                    metrics_handle.flush()
            losses.append(outputs.loss.item())
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}", step=global_step)
            if args.max_steps and global_step >= args.max_steps:
                break

        average_loss = sum(losses) / max(len(losses), 1)
        checkpoint = save_checkpoint(
            model,
            tokenizer,
            args.output_dir,
            epoch,
            global_step,
            average_loss,
        )
        print(f"Checkpoint MLM : {checkpoint}")
        print(f"Loss moyenne   : {average_loss:.4f}")
        if args.max_steps and global_step >= args.max_steps:
            break

    metrics_handle.close()


if __name__ == "__main__":
    main()
