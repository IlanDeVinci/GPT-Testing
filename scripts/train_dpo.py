"""DPO : optimise le modèle SFT sur des préférences (chosen > rejected).

Loss DPO codée à la main, avec un modèle de référence FIGÉ (copie du SFT de départ) :

    L = -log σ( β · [ (logπ(chosen) - logπ_ref(chosen)) - (logπ(rejected) - logπ_ref(rejected)) ] )

Seuls les tokens de la SUITE comptent (le prompt est masqué). À lancer sur GPU, après
avoir construit les paires :

    python scripts/train_dpo.py --pairs data/dpo_pairs.jsonl --resume_checkpoint checkpoints-sft/checkpoint-...
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
from backend.paths import find_latest_checkpoint  # noqa: E402


class DPODataset(Dataset):
    """Chaque exemple = (prompt+chosen, prompt+rejected), avec masque sur la suite."""

    def __init__(self, pairs_path, tokenizer, max_length):
        self.rows = [
            json.loads(line)
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError("Aucune paire DPO. Lance d'abord build_dpo_data.py.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id

    def _encode(self, prompt, continuation):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cont_ids = self.tokenizer.encode(" " + continuation, add_special_tokens=False)
        prefix = [self.bos] if self.bos is not None else []
        ids = prefix + prompt_ids + cont_ids + [self.eos]
        comp = [0] * (len(prefix) + len(prompt_ids)) + [1] * (len(cont_ids) + 1)
        ids = ids[: self.max_length]
        comp = comp[: self.max_length]
        attn = [1] * len(ids)
        padding = self.max_length - len(ids)
        ids += [self.pad] * padding
        attn += [0] * padding
        comp += [0] * padding
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(comp, dtype=torch.long),
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        c_ids, c_attn, c_mask = self._encode(row["prompt"], row["chosen"])
        r_ids, r_attn, r_mask = self._encode(row["prompt"], row["rejected"])
        return {
            "chosen_ids": c_ids, "chosen_attn": c_attn, "chosen_mask": c_mask,
            "rejected_ids": r_ids, "rejected_attn": r_attn, "rejected_mask": r_mask,
        }


def sequence_logprob(model, ids, attention, completion_mask):
    """Somme des log-probabilités sur les tokens de la suite (masque appliqué)."""
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1, :]
    targets = ids[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].float()
    return (token_logp * mask).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO du mini-GPT.")
    parser.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "data" / "dpo_pairs.jsonl")
    parser.add_argument("--resume_checkpoint", type=Path, default=None,
                        help="Checkpoint de base (défaut : dernier SFT dans checkpoints-sft).")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizer-v3" / "tokenizer.json")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "checkpoints-dpo")
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf", action="store_true",
                        help="Base pré-entraînée HF (AutoModel/AutoTokenizer ; défaut : dernier checkpoints-pre-sft).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.hf:
        base_checkpoint = args.resume_checkpoint or find_latest_checkpoint(PROJECT_ROOT / "checkpoints-pre-sft")
        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device)
        reference = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device)
    else:
        tokenizer = load_tokenizer(args.tokenizer, args.max_length)
        base_checkpoint = args.resume_checkpoint or find_latest_checkpoint(PROJECT_ROOT / "checkpoints-sft")
        policy = GPT2LMHeadModel.from_pretrained(base_checkpoint).to(device)
        reference = GPT2LMHeadModel.from_pretrained(base_checkpoint).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    policy.train()

    dataset = DPODataset(args.pairs, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, max(total_updates, 1))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Appareil      : {device}")
    print(f"Paires DPO    : {len(dataset)}")
    print(f"Base (réf+pol): {base_checkpoint.name}")
    print(f"Mises à jour  : {total_updates}")

    metrics_handle = (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8")
    start_time = time.time()
    global_step = 0
    recent_loss = float("nan")
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss, running_count, running_correct, running_pairs = 0.0, 0, 0, 0

    def logps(model, prefix, batch):
        return sequence_logprob(
            model, batch[f"{prefix}_ids"], batch[f"{prefix}_attn"], batch[f"{prefix}_mask"]
        )

    try:
        for epoch in range(1, args.epochs + 1):
            progress = tqdm(dataloader, desc=f"DPO {epoch}/{args.epochs}")
            for batch_index, batch in enumerate(progress, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pi_chosen = logps(policy, "chosen", batch)
                    pi_rejected = logps(policy, "rejected", batch)
                    with torch.no_grad():
                        ref_chosen = logps(reference, "chosen", batch)
                        ref_rejected = logps(reference, "rejected", batch)
                    margin = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
                    loss = -F.logsigmoid(args.beta * margin).mean()

                scaler.scale(loss / args.gradient_accumulation_steps).backward()
                recent_loss = loss.item()
                running_loss += recent_loss
                running_count += 1
                running_correct += (margin > 0).sum().item()
                running_pairs += margin.size(0)

                if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.log_every == 0:
                        pref_acc = running_correct / max(running_pairs, 1)
                        avg_loss = running_loss / max(running_count, 1)
                        metrics_handle.write(
                            json.dumps({
                                "step": global_step, "epoch": epoch,
                                "loss": round(avg_loss, 4),
                                "accuracy": round(pref_acc, 4),
                                "lr": scheduler.get_last_lr()[0],
                                "elapsed_s": round(time.time() - start_time, 1),
                            }) + "\n"
                        )
                        metrics_handle.flush()
                        progress.set_postfix(loss=f"{recent_loss:.3f}", pref_acc=f"{pref_acc:.2f}", step=global_step)
                    else:
                        progress.set_postfix(loss=f"{recent_loss:.3f}", step=global_step)

            save_checkpoint(policy, tokenizer, args.output_dir, epoch, global_step, recent_loss, args)
            print(f"Checkpoint sauvegardé (époque {epoch}, step {global_step})")
        print("DPO terminé.")
    except KeyboardInterrupt:
        print("\nInterruption (Ctrl-C) — sauvegarde...")
        save_checkpoint(policy, tokenizer, args.output_dir, epoch, global_step, recent_loss, args,
                        name=f"checkpoint-interrupted-step-{global_step}")
    finally:
        metrics_handle.close()


if __name__ == "__main__":
    main()
