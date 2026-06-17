"""Entraîne un petit GPT causal from scratch sur le corpus français."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "clean.txt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "checkpoints"


def load_tokenizer(tokenizer_path: Path, model_max_length: int) -> PreTrainedTokenizerFast:
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {tokenizer_path}\n"
            "Lancez d'abord : python scripts/train_tokenizer.py"
        )

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=model_max_length,
    )


def build_token_cache(
    corpus_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    sequence_mode: str,
) -> Path:
    """Tokenise le corpus une seule fois et met les IDs en cache disque (uint16).

    Le cache evite de retokeniser a chaque run et permet une lecture paresseuse par
    memory-mapping, sans charger tout le corpus en RAM.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {corpus_path}\n"
            "Lancez d'abord : python scripts/clean_data.py"
        )

    cache_path = corpus_path.with_name(f"{corpus_path.name}.{sequence_mode}.tokens.bin")
    meta_path = cache_path.with_suffix(".meta.json")
    signature = {
        "corpus_mtime": corpus_path.stat().st_mtime,
        "vocab_size": len(tokenizer),
        "sequence_mode": sequence_mode,
    }
    if cache_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) == signature:
            return cache_path

    if sequence_mode not in ("continuous", "per_line"):
        raise ValueError("sequence_mode doit valoir 'continuous' ou 'per_line'.")
    if len(tokenizer) > 2**16:
        raise ValueError("Vocabulaire trop grand pour un cache uint16.")

    def encode_chunk(buffer: list[str], first: bool) -> list[int]:
        if sequence_mode == "continuous":
            # Le flux est tokenise par morceaux ; on garde le "\n" entre morceaux.
            text = "\n".join(buffer) if first else "\n" + "\n".join(buffer)
            return tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = []
        for row in tokenizer(buffer, add_special_tokens=True)["input_ids"]:
            ids.extend(row)
        return ids

    chunk_lines = 50_000
    tmp_path = cache_path.with_suffix(".bin.tmp")
    n_tokens = 0

    with tmp_path.open("wb") as handle:
        if sequence_mode == "continuous":
            # Un seul BOS/EOS pour tout le flux : le modèle apprend les transitions.
            handle.write(np.asarray([tokenizer.bos_token_id], dtype=np.uint16).tobytes())
            n_tokens += 1

        buffer: list[str] = []
        first = True
        with corpus_path.open("r", encoding="utf-8") as source:
            for raw_line in source:
                line = raw_line.strip()
                if not line:
                    continue
                buffer.append(line)
                if len(buffer) >= chunk_lines:
                    array = np.asarray(encode_chunk(buffer, first), dtype=np.uint16)
                    handle.write(array.tobytes())
                    n_tokens += array.size
                    buffer, first = [], False
        if buffer:
            array = np.asarray(encode_chunk(buffer, first), dtype=np.uint16)
            handle.write(array.tobytes())
            n_tokens += array.size

        if sequence_mode == "continuous":
            handle.write(np.asarray([tokenizer.eos_token_id], dtype=np.uint16).tobytes())
            n_tokens += 1

    if n_tokens < 2:
        tmp_path.unlink(missing_ok=True)
        raise ValueError("Le corpus ne contient pas assez de tokens pour entraîner le modèle.")

    tmp_path.replace(cache_path)
    meta_path.write_text(json.dumps(signature), encoding="utf-8")
    return cache_path


class LazyBlockDataset(Dataset):
    """Lit les tokens en cache par memory-mapping et les decoupe en blocs fixes."""

    def __init__(self, cache_path: Path, block_size: int) -> None:
        self.cache_path = cache_path
        self.block_size = block_size
        n_tokens = cache_path.stat().st_size // 2  # uint16 = 2 octets
        self.n_blocks = (n_tokens - 1) // block_size
        if self.n_blocks < 1:
            raise ValueError("Le corpus ne contient pas assez de tokens pour un bloc.")
        self._tokens: np.memmap | None = None

    @property
    def tokens(self) -> np.memmap:
        # Ouverture paresseuse : chaque worker DataLoader mappe le fichier lui-meme.
        if self._tokens is None:
            self._tokens = np.memmap(self.cache_path, dtype=np.uint16, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        chunk = np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64)
        input_ids = torch.from_numpy(chunk)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def token_accuracy(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Exactitude de prediction du token suivant sur le batch courant."""
    predictions = logits[:, :-1, :].argmax(dim=-1)
    targets = input_ids[:, 1:]
    return (predictions == targets).float().mean().item()


def save_checkpoint(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    output_root: Path,
    epoch: int,
    global_step: int,
    loss: float,
    args: argparse.Namespace,
    name: str | None = None,
    trainer_state: dict | None = None,
) -> Path:
    base_name = name or f"checkpoint-epoch-{epoch}-step-{global_step}"
    checkpoint_dir = output_root / base_name
    suffix = 2
    while checkpoint_dir.exists():
        checkpoint_dir = output_root / f"{base_name}-run-{suffix}"
        suffix += 1
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "average_loss": loss,
        "training_arguments": vars(args),
    }
    # Les objets Path ne sont pas directement sérialisables en JSON.
    metadata["training_arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["training_arguments"].items()
    }
    (checkpoint_dir / "training_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint_dir.name}, indent=2),
        encoding="utf-8",
    )
    # Etat complet (optimiseur, scheduler, scaler, step) pour une reprise sans couture.
    if trainer_state is not None:
        torch.save(trainer_state, checkpoint_dir / "training_state.pt")
    return checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne un mini GPT from scratch.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument(
        "--sequence_mode",
        choices=["continuous", "per_line"],
        default="continuous",
        help="continuous apprend les transitions entre lignes (recommandé).",
    )
    parser.add_argument("--n_positions", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Arrête après N mises à jour; 0 entraîne pendant toutes les époques.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Frequence (en updates) d'ecriture des metriques pour la visualisation.",
    )
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--fresh_optimizer",
        action="store_true",
        help="Reprend les POIDS du checkpoint mais repart avec un optimiseur et "
        "un scheduler neufs (pour l'adaptation de domaine à faible LR).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Réduit la VRAM en recalculant des activations pendant le backward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size > args.n_positions:
        raise ValueError("--block_size doit être inférieur ou égal à --n_positions.")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n_embd doit être divisible par --n_head.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps doit être au moins égal à 1.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # TF32 : matmuls ~1.5x plus rapides sur Ampere+ (RTX 40xx) sans perte
        # de qualité notable pour l'entraînement.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer, args.n_positions)
    cache_path = build_token_cache(args.corpus, tokenizer, args.sequence_mode)
    dataset = LazyBlockDataset(cache_path, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # SDPA (scaled_dot_product_attention) : ~3x plus rapide que l'attention
    # "eager" sur ce GPU, sans changer les résultats.
    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    if args.resume_checkpoint is not None:
        model = GPT2LMHeadModel.from_pretrained(
            args.resume_checkpoint, attn_implementation=attn_impl
        ).to(device)
        if model.config.vocab_size != len(tokenizer):
            raise ValueError("Le checkpoint et le tokenizer n'ont pas le même vocabulaire.")
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=args.n_positions,
            n_ctx=args.n_positions,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            loss_type="ForCausalLM",
            attn_implementation=attn_impl,
        )
        model = GPT2LMHeadModel(config).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    # Transformers 4.55 lit cet attribut sur le modèle pour choisir la loss.
    model.loss_type = "ForCausalLM"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    planned_updates = updates_per_epoch * args.epochs
    total_updates = (
        min(planned_updates, args.max_steps) if args.max_steps > 0 else planned_updates
    )
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_updates, 1),
    )

    use_amp = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reprise complete : si le checkpoint contient l'etat d'entrainement, on restaure
    # optimiseur + scheduler + scaler + step pour continuer sans couture (LR continu).
    resume_step = 0
    if args.resume_checkpoint is not None and args.fresh_optimizer:
        print("Adaptation de domaine : poids repris, optimiseur + scheduler NEUFS.")
    elif args.resume_checkpoint is not None:
        state_file = args.resume_checkpoint / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            total_updates = state["total_updates"]
            warmup_steps = state["warmup_steps"]
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max(total_updates, 1),
            )
            optimizer.load_state_dict(state["optimizer"])
            # Remet l'etat de l'optimiseur sur le bon peripherique (robuste GPU/CPU).
            for opt_state in optimizer.state.values():
                for key, value in opt_state.items():
                    if isinstance(value, torch.Tensor):
                        opt_state[key] = value.to(device)
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            resume_step = state["global_step"]
            print(f"Reprise complete : optimiseur + scheduler restaures.")
        else:
            print("Reprise des poids seulement (checkpoint sans etat d'entrainement).")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Appareil              : {device}")
    print(f"Blocs d'entraînement  : {len(dataset)}")
    print(f"Tokens par bloc       : {args.block_size}")
    print(f"Paramètres du modèle  : {parameter_count:,}")
    print(f"Mises à jour (plan)   : {total_updates}  (reprise au step {resume_step})")

    # On n'ajoute aux métriques existantes que si la reprise continue le compteur
    # d'étapes (reprise complète). Un warm-restart (step remis à 0) repart d'un fichier neuf.
    metrics_mode = "a" if resume_step > 0 else "w"
    metrics_handle = (args.output_dir / "metrics.jsonl").open(metrics_mode, encoding="utf-8")
    start_time = time.time()
    global_step = resume_step
    recent_loss = float("nan")
    epoch = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            batches_seen = 0
            progress = tqdm(dataloader, desc=f"Époque {epoch}/{args.epochs}")

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
                        accuracy = token_accuracy(outputs.logits, batch["input_ids"])
                        metrics_handle.write(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "epoch": epoch,
                                    "loss": round(recent_loss, 4),
                                    "accuracy": round(accuracy, 4),
                                    "lr": scheduler.get_last_lr()[0],
                                    "elapsed_s": round(time.time() - start_time, 1),
                                }
                            )
                            + "\n"
                        )
                        metrics_handle.flush()
                        progress.set_postfix(
                            loss=f"{recent_loss:.3f}",
                            acc=f"{accuracy:.3f}",
                            step=global_step,
                        )
                    else:
                        progress.set_postfix(loss=f"{recent_loss:.3f}", step=global_step)

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break

            average_loss = epoch_loss / max(batches_seen, 1)
            checkpoint = save_checkpoint(
                model, tokenizer, args.output_dir, epoch, global_step, average_loss, args,
                trainer_state={
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": global_step,
                    "total_updates": total_updates,
                    "warmup_steps": warmup_steps,
                },
            )
            print(f"Checkpoint sauvegardé : {checkpoint}")
            print(f"Loss moyenne           : {average_loss:.4f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        print("Entraînement terminé.")
    except KeyboardInterrupt:
        print("\nInterruption (Ctrl-C) — sauvegarde du modèle en cours...")
        checkpoint = save_checkpoint(
            model,
            tokenizer,
            args.output_dir,
            epoch,
            global_step,
            recent_loss,
            args,
            name=f"checkpoint-interrupted-step-{global_step}",
            trainer_state={
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "total_updates": total_updates,
                "warmup_steps": warmup_steps,
            },
        )
        print(f"Checkpoint d'interruption sauvegardé : {checkpoint}")
    finally:
        metrics_handle.close()


if __name__ == "__main__":
    main()
