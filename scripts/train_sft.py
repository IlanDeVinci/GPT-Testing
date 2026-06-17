"""Fine-tuning supervisé (SFT) du mini-GPT : apprendre à CONTINUER une histoire.

On construit des paires (début -> suite) à partir de phrases consécutives du corpus
narratif, et on n'entraîne la loss QUE sur la suite (les tokens du début ont un label
-100, donc ignorés). Le modèle apprend ainsi à prolonger un début de façon cohérente,
au lieu de dériver vers le style encyclopédique de son pré-entraînement.

À lancer APRÈS avoir une base entraînée (on reprend ses poids) :

    python scripts/train_sft.py --resume_checkpoint checkpoints-v3/checkpoint-epoch-1-step-6500
"""

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
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model import load_tokenizer, save_checkpoint  # noqa: E402


class SFTDataset(Dataset):
    """Paires (début -> suite). Loss uniquement sur la suite (début = labels -100)."""

    def __init__(
        self,
        corpus_path,
        tokenizer,
        max_length,
        cont_sentences,
        max_examples,
        seed,
    ):
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus introuvable : {corpus_path}")
        self.lines = [
            line.strip()
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cont_sentences = cont_sentences
        available = max(0, len(self.lines) - cont_sentences)
        indices = list(range(available))
        if max_examples > 0 and max_examples < available:
            rng = random.Random(seed)
            rng.shuffle(indices)
            indices = indices[:max_examples]
        self.indices = indices
        self.length = len(indices)
        if self.length < 1:
            raise ValueError("Corpus trop court pour construire des paires SFT.")
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line_index = self.indices[index]
        prompt = self.lines[line_index]
        continuation = " ".join(
            self.lines[line_index + 1 : line_index + 1 + self.cont_sentences]
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cont_ids = self.tokenizer.encode(" " + continuation, add_special_tokens=False)

        prefix = [self.bos] if self.bos is not None else []
        ids = prefix + prompt_ids + cont_ids + [self.eos]
        labels = [-100] * (len(prefix) + len(prompt_ids)) + cont_ids + [self.eos]
        ids = ids[: self.max_length]
        labels = labels[: self.max_length]

        attention = [1] * len(ids)
        padding = self.max_length - len(ids)
        ids += [self.pad] * padding
        attention += [0] * padding
        labels += [-100] * padding

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def completion_metrics(logits, labels) -> dict[str, float]:
    """Métriques sur la suite uniquement.

    `top1_accuracy` est la métrique stricte classique : le prochain token exact
    doit être premier. Pour suivre un SFT narratif, elle est très sévère car
    plusieurs mots peuvent être plausibles. `top5_accuracy` est donc exposée comme
    `accuracy` principale : le bon token doit être dans les 5 choix les plus
    probables, ce qui donne une lecture plus pédagogique et plus stable.
    """
    shifted_logits = logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    if mask.sum() == 0:
        return {
            "accuracy": 0.0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
        }

    valid_logits = shifted_logits[mask]
    valid_targets = targets[mask]
    top_k = min(10, valid_logits.size(-1))
    top_predictions = valid_logits.topk(k=top_k, dim=-1).indices
    top1 = (top_predictions[:, 0] == valid_targets).float().mean().item()
    top5_width = min(5, top_k)
    top5 = (
        top_predictions[:, :top5_width]
        .eq(valid_targets.unsqueeze(-1))
        .any(dim=-1)
        .float()
        .mean()
        .item()
    )
    top10 = (
        top_predictions.eq(valid_targets.unsqueeze(-1))
        .any(dim=-1)
        .float()
        .mean()
        .item()
    )
    return {
        "accuracy": top5,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "top10_accuracy": top10,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT du mini-GPT (continuation).")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "clean-narratif.txt")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizer-v3" / "tokenizer.json")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "checkpoints-sft")
    parser.add_argument("--resume_checkpoint", type=Path, default=None,
                        help="Base from-scratch : un checkpoint local.")
    parser.add_argument("--hf_model", type=str, default=None,
                        help="Base pré-entraînée Hugging Face, ex. asi/gpt-fr-cased-small.")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--cont_sentences", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--time_budget_minutes",
        type=float,
        default=0,
        help="Arrête proprement et sauvegarde après ce nombre de minutes.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.hf_model:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.hf_model).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        base_label = args.hf_model
        lineage = "pretrained"
        print(f"Base pré-entraînée : {base_label}")
    elif args.resume_checkpoint:
        tokenizer = load_tokenizer(args.tokenizer, args.max_length)
        model = GPT2LMHeadModel.from_pretrained(args.resume_checkpoint).to(device)
        if model.config.vocab_size != len(tokenizer):
            raise ValueError("Le checkpoint et le tokenizer n'ont pas le même vocabulaire.")
        base_label = args.resume_checkpoint.name
        lineage = "from-scratch"
    else:
        raise SystemExit("Fournis --hf_model (pré-entraîné) ou --resume_checkpoint (from-scratch).")
    args.model_lineage = lineage
    args.base_model = base_label
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.loss_type = "ForCausalLM"

    dataset = SFTDataset(
        args.corpus,
        tokenizer,
        args.max_length,
        args.cont_sentences,
        args.max_examples,
        args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    planned = updates_per_epoch * args.epochs
    total_updates = min(planned, args.max_steps) if args.max_steps > 0 else planned
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max(total_updates, 1))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Appareil          : {device}")
    print(f"Paires SFT        : {len(dataset)}")
    print(f"Base              : {base_label}")
    print(f"Lignage           : {lineage}")
    print(f"Mises à jour      : {total_updates}")
    if args.time_budget_minutes > 0:
        print(f"Budget temps      : {args.time_budget_minutes:.1f} min")

    metrics_handle = (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8")
    start_time = time.time()
    global_step = 0
    recent_loss = float("nan")
    epoch = 0
    stop_for_time = False
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            batches_seen = 0
            progress = tqdm(dataloader, desc=f"SFT {epoch}/{args.epochs}")
            for batch_index, batch in enumerate(progress, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(**batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                epoch_loss += raw_loss.item()
                batches_seen += 1

                should_update = (
                    batch_index % args.gradient_accumulation_steps == 0
                    or batch_index == len(dataloader)
                )
                if should_update:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    recent_loss = raw_loss.item()

                    if global_step % args.log_every == 0:
                        metrics = completion_metrics(outputs.logits, batch["labels"])
                        metrics_handle.write(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "epoch": epoch,
                                    "loss": round(recent_loss, 4),
                                    "accuracy": round(metrics["accuracy"], 4),
                                    "top1_accuracy": round(metrics["top1_accuracy"], 4),
                                    "top5_accuracy": round(metrics["top5_accuracy"], 4),
                                    "top10_accuracy": round(metrics["top10_accuracy"], 4),
                                    "lr": scheduler.get_last_lr()[0],
                                    "elapsed_s": round(time.time() - start_time, 1),
                                }
                            )
                            + "\n"
                        )
                        metrics_handle.flush()
                        progress.set_postfix(
                            loss=f"{recent_loss:.3f}",
                            top5=f"{metrics['top5_accuracy']:.3f}",
                            top1=f"{metrics['top1_accuracy']:.3f}",
                            step=global_step,
                        )
                    else:
                        progress.set_postfix(loss=f"{recent_loss:.3f}", step=global_step)

                    if (
                        args.time_budget_minutes > 0
                        and time.time() - start_time >= args.time_budget_minutes * 60
                    ):
                        stop_for_time = True
                        break

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break

            average_loss = epoch_loss / max(batches_seen, 1)
            checkpoint = save_checkpoint(
                model, tokenizer, args.output_dir, epoch, global_step, average_loss, args
            )
            print(f"Checkpoint sauvegardé : {checkpoint}")
            print(f"Loss moyenne           : {average_loss:.4f}")
            if stop_for_time:
                print("Arrêt propre : budget temps atteint.")
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        print("SFT terminé.")
    except KeyboardInterrupt:
        print("\nInterruption (Ctrl-C) — sauvegarde du modèle en cours...")
        checkpoint = save_checkpoint(
            model, tokenizer, args.output_dir, epoch, global_step, recent_loss, args,
            name=f"checkpoint-interrupted-step-{global_step}",
        )
        print(f"Checkpoint d'interruption sauvegardé : {checkpoint}")
    finally:
        metrics_handle.close()


if __name__ == "__main__":
    main()
