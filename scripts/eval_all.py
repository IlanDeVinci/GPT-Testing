"""Evalue TOUS les checkpoints GPT du projet sur un corpus de validation commun.

Pourquoi un script dedie plutot que evaluate_perplexity.py lance 5 fois :
- les deux lignages n'utilisent pas le meme tokenizer (BPE maison 16k vs celui
  d'asi/gpt-fr-cased-small). La perplexite *par token* n'est donc PAS comparable
  d'un lignage a l'autre (elle depend du decoupage). On calcule en plus une
  perplexite *par mot* (exp(NLL_total / nb_mots)) qui, elle, est comparable.
- on evalue exactement le meme TEXTE (les premiers --max_chars du corpus de val)
  et le meme block_size pour tout le monde -> comparaison honnete.

    python scripts/eval_all.py --val_corpus data/clean-v3-val.txt

Sorties : tableau console + outputs/eval_all.json + outputs/eval_all.png.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model import load_tokenizer, token_accuracy  # noqa: E402
from backend.paths import find_latest_checkpoint  # noqa: E402


# (id, libelle, lignage, dossier de checkpoints)
MODELS = [
    ("v3", "From-scratch - pre-entraine", "from-scratch", "checkpoints-v3"),
    ("v4", "From-scratch - v4 (~34M)", "from-scratch", "checkpoints-v4"),
    ("narratif", "From-scratch - specialise narratif", "from-scratch", "checkpoints-narratif"),
    ("sft", "From-scratch - SFT", "from-scratch", "checkpoints-sft"),
    ("dpo", "From-scratch - DPO", "from-scratch", "checkpoints-dpo"),
    ("pre-sft", "Pre-entraine FR - SFT", "pretrained", "checkpoints-pre-sft"),
    ("pre-dpo", "Pre-entraine FR - DPO", "pretrained", "checkpoints-pre-dpo"),
]


def load_model_and_tokenizer(lineage: str, checkpoint: Path, tokenizer_path: Path, device):
    """Charge proprement (low_cpu_mem_usage=False + tie_weights pour eviter le
    bug 'meta tensor' des poids lies absents du safetensors)."""
    if lineage == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=False)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        model = GPT2LMHeadModel.from_pretrained(checkpoint, low_cpu_mem_usage=False)
        ctx = model.config.n_positions
        tokenizer = load_tokenizer(tokenizer_path, ctx)
    model.tie_weights()
    model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_one(model, tokenizer, text: str, block_size: int, batch_size: int, device) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_blocks = len(ids) // block_size
    if n_blocks < 1:
        raise ValueError("Texte trop court pour un seul bloc.")
    ids = ids[: n_blocks * block_size]

    # Mots reellement evalues (pour la perplexite par mot, comparable inter-tokenizers).
    n_words = len(tokenizer.decode(ids).split())

    blocks = torch.tensor(ids, dtype=torch.long, device=device).view(n_blocks, block_size)
    use_amp = device.type == "cuda"
    total_nll = 0.0          # somme des log-vraisemblances negatives (nats)
    total_pred_tokens = 0    # tokens reellement predits (block_size-1 par sequence)
    acc_weighted = 0.0
    seqs = 0
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            batch_ids = blocks[start : start + batch_size]
            inputs = {
                "input_ids": batch_ids,
                "attention_mask": torch.ones_like(batch_ids),
                "labels": batch_ids,
            }
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)
            b = batch_ids.shape[0]
            pred = b * (block_size - 1)        # loss HF moyennee sur ces positions
            total_nll += outputs.loss.item() * pred
            total_pred_tokens += pred
            acc_weighted += token_accuracy(outputs.logits, batch_ids) * b
            seqs += b

    mean_token_loss = total_nll / total_pred_tokens
    return {
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "blocks": n_blocks,
        "words_eval": n_words,
        "token_loss": round(mean_token_loss, 4),
        "token_perplexity": round(math.exp(mean_token_loss), 2),
        "word_perplexity": round(math.exp(total_nll / n_words), 2),
        "next_token_accuracy": round(acc_weighted / seqs, 4),
    }


def bar_chart(results: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    ok = [r for r in results if "error" not in r]
    labels = [r["label"] for r in ok]
    word_ppl = [r["word_perplexity"] for r in ok]
    acc = [r["next_token_accuracy"] for r in ok]
    colors = ["#534ab7" if r["lineage"] == "from-scratch" else "#0f6e56" for r in ok]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    y = range(len(labels))
    ax1.barh(list(y), word_ppl, color=colors)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("perplexite par mot (plus bas = mieux)")
    ax1.set_title("Perplexite par mot - comparable entre lignages")
    for i, v in enumerate(word_ppl):
        ax1.text(v, i, f" {v:.0f}", va="center", fontsize=9)

    ax2.barh(list(y), acc, color=colors)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([])
    ax2.invert_yaxis()  # meme ordre vertical que le panneau de gauche
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("accuracy token suivant (plus haut = mieux)")
    ax2.set_title("Accuracy de prediction")
    for i, v in enumerate(acc):
        ax2.text(v, i, f" {v:.3f}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color="#534ab7", label="from-scratch"), Patch(color="#0f6e56", label="pre-entraine FR")],
        loc="lower center", ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nGraphe comparatif sauvegarde : {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalue tous les checkpoints GPT.")
    parser.add_argument("--val_corpus", type=Path, default=Path("data/clean-v3-val.txt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer-v3/tokenizer.json"),
                        help="Tokenizer du lignage from-scratch.")
    parser.add_argument("--block_size", type=int, default=256, help="Identique pour tous (<= ctx mini).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_chars", type=int, default=1_200_000,
                        help="Longueur de texte de val evaluee (meme texte pour tous).")
    parser.add_argument("--out_json", type=Path, default=Path("outputs/eval_all.json"))
    parser.add_argument("--out_png", type=Path, default=Path("outputs/eval_all.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = args.val_corpus.read_text(encoding="utf-8")[: args.max_chars]
    print(f"Appareil : {device} | texte de val : {len(text):,} caracteres, {len(text.split()):,} mots\n")

    results: list[dict] = []
    for model_id, label, lineage, dir_name in MODELS:
        root = PROJECT_ROOT / dir_name
        entry = {"id": model_id, "label": label, "lineage": lineage, "dir": dir_name}
        if not (root / "latest.json").exists() and not list(root.glob("checkpoint-*")):
            entry["error"] = "checkpoint absent"
            results.append(entry)
            print(f"[skip] {label:32s} : pas de checkpoint")
            continue
        try:
            checkpoint = find_latest_checkpoint(root)
            model, tokenizer = load_model_and_tokenizer(lineage, checkpoint, args.tokenizer, device)
            metrics = evaluate_one(model, tokenizer, text, args.block_size, args.batch_size, device)
            entry.update({"checkpoint": checkpoint.name, **metrics})
            print(
                f"[ok]   {label:32s} : ppl/mot={metrics['word_perplexity']:8.1f} "
                f"ppl/token={metrics['token_perplexity']:7.1f} acc={metrics['next_token_accuracy']:.3f} "
                f"({metrics['params_millions']}M params)"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # on continue meme si un modele echoue
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[err]  {label:32s} : {entry['error']}")
        results.append(entry)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultats JSON : {args.out_json}")
    if any("word_perplexity" in r for r in results):
        bar_chart(results, args.out_png)


if __name__ == "__main__":
    main()
