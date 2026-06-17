"""Récompense portée par le mini-BERT (MLM) : plausibilité d'un texte.

On masque régulièrement ~20 % des tokens et on moyenne la log-probabilité que BERT
attribue aux vrais tokens. Plus c'est haut, plus le texte est plausible/fluide pour
BERT. En scorant « prompt + suite », une suite cohérente avec le prompt obtient un
meilleur score (BERT prédit les tokens masqués en s'appuyant sur le contexte du début).

C'est le « juge » du DPO : il remplace une note humaine pour classer les suites.
"""

import sys
from pathlib import Path

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.paths import BERT_MLM_DIR, BERT_TOKENIZER_DIR, find_latest_checkpoint  # noqa: E402


class BertJudge:
    """Score un texte par pseudo-log-vraisemblance du mini-BERT MLM."""

    def __init__(self, mlm_dir: Path = BERT_MLM_DIR, tokenizer_dir: Path = BERT_TOKENIZER_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        self.checkpoint = find_latest_checkpoint(mlm_dir)
        self.model = BertForMaskedLM.from_pretrained(self.checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str, mask_frac: float = 0.2) -> float:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )
        ids = encoded["input_ids"][0]
        specials = set(self.tokenizer.all_special_ids)
        positions = [i for i, token in enumerate(ids.tolist()) if token not in specials]
        if len(positions) < 2:
            return 0.0

        step = max(1, round(1 / mask_frac))
        masked_positions = positions[::step]
        masked = ids.clone()
        masked[masked_positions] = self.tokenizer.mask_token_id

        logits = self.model(
            input_ids=masked.unsqueeze(0).to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
        ).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        total = sum(log_probs[pos, ids[pos]].item() for pos in masked_positions)
        return total / len(masked_positions)
