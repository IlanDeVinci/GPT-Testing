"""Remplissage de trous avec le mini-BERT MLM entraîné localement."""

from pathlib import Path

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

from backend.paths import BERT_MLM_DIR, BERT_TOKENIZER_DIR, find_latest_checkpoint


class MaskFiller:
    """Charge le mini-BERT MLM et propose des mots pour un seul [MASK]."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        tokenizer_dir: Path = BERT_TOKENIZER_DIR,
    ) -> None:
        checkpoint = checkpoint or find_latest_checkpoint(BERT_MLM_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        # low_cpu_mem_usage=False : matérialise tous les poids en RAM avant
        # le .to(cuda). Les checkpoints safetensors ne stockent pas le poids
        # lié cls.predictions.decoder.weight ; tie_weights() le relie à
        # l'embedding pour éviter le "Cannot copy out of meta tensor".
        self.model = BertForMaskedLM.from_pretrained(
            checkpoint, low_cpu_mem_usage=False
        )
        self.model.tie_weights()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.checkpoint = checkpoint

    def suggest(self, sentence: str, top_k: int = 5) -> list[dict]:
        sentence = sentence.replace("[MASK]", self.tokenizer.mask_token)
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.device)

        positions = (
            encoded["input_ids"][0] == self.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError("La phrase ne contient pas de [MASK].")

        index = positions[0].item()
        with torch.no_grad():
            logits = self.model(**encoded).logits[0, index]
        probabilities = torch.softmax(logits, dim=-1)

        top = torch.topk(probabilities, min(top_k * 6, probabilities.shape[-1]))
        suggestions: list[dict] = []
        seen: set[str] = set()
        for score, token_id in zip(top.values.tolist(), top.indices.tolist()):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("##") or not token.isalpha() or len(token) < 2:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            suggestions.append({"word": token, "score": round(score, 4)})
            if len(suggestions) >= top_k:
                break
        return suggestions
