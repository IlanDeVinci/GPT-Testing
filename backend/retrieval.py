"""Récupération sémantique (RAG) bâtie sur NOTRE mini-BERT — aucun poids externe.

Le mini-BERT entraîné en MLM sert d'encodeur de phrases : on moyenne ses états
cachés (mean-pooling masqué) pour obtenir un vecteur par passage, puis on cherche
par similarité cosinus. Aucune dépendance à sentence-transformers : l'encodeur
est le modèle maison, ce qui reste cohérent avec l'esprit "from scratch".

Deux usages dans le jeu :
- conditionnement de style : on récupère des passages proches du début et on les
  met en contexte du mini-GPT ;
- filet de sécurité : on propose une vraie continuation humaine du corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, PreTrainedTokenizerFast

from backend.paths import BERT_MLM_DIR, BERT_TOKENIZER_DIR, DATA_DIR, find_latest_checkpoint

RAG_INDEX_DIR = DATA_DIR / "rag-index"


def load_bert_encoder(checkpoint: Path | None = None):
    """Charge le mini-BERT comme encodeur (sans la tête MLM) + son tokenizer."""
    checkpoint = checkpoint or find_latest_checkpoint(BERT_MLM_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(BERT_TOKENIZER_DIR)
    # BertModel ignore proprement les poids de la tête MLM du checkpoint.
    model = BertModel.from_pretrained(checkpoint, low_cpu_mem_usage=False).to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def embed_texts(
    model: BertModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """Embeddings par mean-pooling masqué, shape [N, hidden].

    normalize=True : L2-normalisé (cosinus direct). normalize=False : vecteurs
    bruts, pour pouvoir d'abord les centrer (réduit l'anisotropie du BERT MLM).
    """
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state  # [B, T, H]
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def center_normalize(vectors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Soustrait la moyenne du corpus puis L2-normalise (« all-but-the-mean »)."""
    centered = vectors - mean
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / np.clip(norms, 1e-9, None)


class Retriever:
    """Charge l'index pré-calculé et répond aux requêtes top-k."""

    def __init__(self, index_dir: Path = RAG_INDEX_DIR) -> None:
        emb_path = index_dir / "embeddings.npy"
        passages_path = index_dir / "passages.txt"
        if not emb_path.exists() or not passages_path.exists():
            raise FileNotFoundError(
                f"Index RAG introuvable dans {index_dir}. "
                "Construis-le d'abord : python scripts/build_rag_index.py"
            )
        # Stocké en float16 pour l'espace disque ; on repasse en float32 pour le calcul.
        # Embeddings de l'index : déjà centrés + normalisés à la construction.
        self.embeddings = np.load(emb_path).astype(np.float32)
        self.mean = np.load(index_dir / "mean.npy").astype(np.float32)
        self.passages = passages_path.read_text(encoding="utf-8").splitlines()
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.model, self.tokenizer, self.device = load_bert_encoder()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        raw = embed_texts(self.model, self.tokenizer, [query], self.device, normalize=False)
        query_vec = center_normalize(raw, self.mean)[0]
        scores = self.embeddings @ query_vec  # cosinus (vecteurs centrés + normalisés)
        top = np.argsort(-scores)[:top_k]
        return [
            {"text": self.passages[i], "score": round(float(scores[i]), 4)}
            for i in top
        ]
