"""Chargement et inférence du mini-GPT — autonome (vendoré depuis l'ancien jeu)."""

from pathlib import Path
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from backend.paths import (
    FROMSCRATCH_DIR,
    GPT_CHECKPOINTS_DIR,
    GPT_IS_PRETRAINED,
    GPT_TOKENIZER_PATH,
    PRETRAINED_DIR,
    find_latest_checkpoint,
)


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENTENCE_END_RE = re.compile(r"[.!?…]+(?:[\"'»)\]]+)?")


def model_context_size(model: torch.nn.Module, tokenizer: object) -> int:
    """Retourne la taille de contexte, compatible GPT maison et modèles HF."""
    config = model.config
    for name in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return tokenizer_limit
    return 256


def trim_to_complete_sentence(text: str, min_words: int = 8) -> str:
    """Retire le fragment final si au moins une phrase utile est terminée."""
    endings = list(SENTENCE_END_RE.finditer(text))
    if not endings:
        return text
    completed = text[: endings[-1].end()].strip()
    if len(completed.split()) < min_words:
        return text
    return completed


def load_gpt_components(
    checkpoint: Path | None = None,
    tokenizer_path: Path = GPT_TOKENIZER_PATH,
    is_pretrained: bool | None = None,
):
    if is_pretrained is None:
        is_pretrained = GPT_IS_PRETRAINED
    checkpoint = checkpoint or find_latest_checkpoint(GPT_CHECKPOINTS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_pretrained:
        # Lignage pré-entraîné : modèle ET tokenizer chargés depuis le checkpoint.
        # low_cpu_mem_usage=False + tie_weights() : voir mask_fill.py. Les
        # safetensors ne stockent pas lm_head.weight (lié à wte) ; sans ça le
        # .to(cuda) plante par intermittence ("Cannot copy out of meta tensor").
        model = AutoModelForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=False)
        model.tie_weights()
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # Lignage from-scratch : GPT-2 maison + tokenizer BPE local.
        model = GPT2LMHeadModel.from_pretrained(checkpoint, low_cpu_mem_usage=False)
        model.tie_weights()
        model = model.to(device)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            model_max_length=model.config.n_positions,
        )

    model.eval()
    tokenizer.truncation_side = "left"
    return model, tokenizer, device, checkpoint


class GPTStoryGenerator:
    """Produit uniquement la nouvelle continuation, sans répéter le contexte."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        tokenizer_path: Path = GPT_TOKENIZER_PATH,
        is_pretrained: bool | None = None,
    ) -> None:
        self.model, self.tokenizer, self.device, self.checkpoint = load_gpt_components(
            checkpoint,
            tokenizer_path,
            is_pretrained,
        )

    def generate(
        self,
        context: str,
        num_candidates: int = 3,
        max_new_tokens: int = 40,
        temperature: float = 0.85,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        min_new_tokens: int = 12,
        min_chars: int = 60,
        seed: int = 42,
    ) -> list[str]:
        prompt_ids = self.tokenizer.encode(context.strip(), add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            prompt_ids = [self.tokenizer.bos_token_id] + prompt_ids
        max_positions = model_context_size(self.model, self.tokenizer)
        if len(prompt_ids) >= max_positions:
            prompt_ids = prompt_ids[-max_positions:]

        available = max_positions - len(prompt_ids)
        max_new_tokens = min(max_new_tokens, available)
        if max_new_tokens < 1:
            raise ValueError("Le contexte occupe toute la fenêtre du mini-GPT.")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        continuations: list[str] = []
        too_short: list[str] = []
        min_new_tokens = min(min_new_tokens, max_new_tokens)
        for attempt in range(8):
            missing = num_candidates - len(continuations)
            if missing <= 0:
                break
            torch.manual_seed(seed + attempt)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + attempt)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    num_return_sequences=missing,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_length = input_ids.shape[1]
            for sequence in outputs:
                generated_ids = sequence[prompt_length:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                text = text.replace("�", " [UNK] ")
                text = CONTROL_RE.sub(" ", text)
                for token in self.tokenizer.all_special_tokens:
                    if token != self.tokenizer.unk_token:
                        text = text.replace(token, "")
                cleaned = " ".join(text.split()).strip()
                cleaned = trim_to_complete_sentence(cleaned)
                if not cleaned or cleaned in continuations or cleaned in too_short:
                    continue
                if len(cleaned) < min_chars:
                    too_short.append(cleaned)
                    continue
                continuations.append(cleaned)

        if len(continuations) < num_candidates:
            # Complète avec les plus longues continuations rejetées, des plus
            # longues aux plus courtes, plutôt que d'échouer franchement.
            for fallback in sorted(too_short, key=len, reverse=True):
                if len(continuations) >= num_candidates:
                    break
                continuations.append(fallback)

        if len(continuations) < num_candidates:
            raise RuntimeError(
                "Le mini-GPT n'a pas produit assez de continuations distinctes. "
                "Réentraînez-le en mode continu ou augmentez la température."
            )
        return continuations[:num_candidates]


def make_generator(lineage: str = "from-scratch") -> GPTStoryGenerator:
    """Charge le générateur d'un lignage : 'from-scratch' (défaut) ou 'pretrained'."""
    if lineage == "pretrained":
        if PRETRAINED_DIR is None:
            raise FileNotFoundError("Aucun modèle pré-entraîné (dossiers checkpoints-pre-*).")
        return GPTStoryGenerator(
            checkpoint=find_latest_checkpoint(PRETRAINED_DIR), is_pretrained=True
        )
    return GPTStoryGenerator(
        checkpoint=find_latest_checkpoint(FROMSCRATCH_DIR), is_pretrained=False
    )
