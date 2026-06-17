"""Construit l'index RAG : embeddings du corpus narratif via NOTRE mini-BERT.

Sortie dans data/rag-index/ :
- embeddings.npy : matrice [N, hidden] en float16 (gain d'espace) ;
- passages.txt   : les N passages (un par ligne, même ordre) ;
- meta.json      : checkpoint encodeur, dimension, nombre de passages.

    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --corpus data/narratif-train.txt --max_passages 200000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.paths import BERT_MLM_DIR, find_latest_checkpoint  # noqa: E402
from backend.retrieval import (  # noqa: E402
    RAG_INDEX_DIR,
    center_normalize,
    embed_texts,
    load_bert_encoder,
)


def read_passages(corpus: Path, min_chars: int, max_passages: int) -> list[str]:
    """Une ligne = un passage (le corpus est déjà nettoyé phrase par phrase)."""
    passages: list[str] = []
    seen: set[str] = set()
    with corpus.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if len(line) < min_chars or line in seen:
                continue
            seen.add(line)
            passages.append(line)
            if max_passages and len(passages) >= max_passages:
                break
    return passages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit l'index RAG (mini-BERT).")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "narratif-train.txt")
    parser.add_argument("--output_dir", type=Path, default=RAG_INDEX_DIR)
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--max_passages", type=int, default=200_000,
                        help="Plafond pour garder l'index léger (0 = tout).")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise SystemExit(
            f"Corpus introuvable : {args.corpus}. Construis d'abord le corpus narratif "
            "(scripts/clean_data.py ... --output data/narratif-train.txt)."
        )

    passages = read_passages(args.corpus, args.min_chars, args.max_passages)
    if not passages:
        raise SystemExit("Aucun passage retenu.")
    print(f"Passages à indexer : {len(passages):,}")

    model, tokenizer, device = load_bert_encoder()
    checkpoint = find_latest_checkpoint(BERT_MLM_DIR)
    print(f"Encodeur           : {checkpoint.name} sur {device}")

    raw = embed_texts(model, tokenizer, passages, device, batch_size=args.batch_size, normalize=False)
    mean = raw.mean(axis=0, keepdims=True)
    embeddings = center_normalize(raw, mean)  # centrage « all-but-the-mean » + L2
    print(f"Embeddings         : {embeddings.shape} (dim {embeddings.shape[1]}, centrés)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings.astype(np.float16))
    np.save(args.output_dir / "mean.npy", mean.astype(np.float32))
    (args.output_dir / "passages.txt").write_text("\n".join(passages), encoding="utf-8")
    (args.output_dir / "meta.json").write_text(
        json.dumps(
            {
                "encoder_checkpoint": str(checkpoint),
                "hidden_size": int(embeddings.shape[1]),
                "n_passages": len(passages),
                "corpus": str(args.corpus),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    size_mb = (args.output_dir / "embeddings.npy").stat().st_size / 1e6
    print(f"Index écrit        : {args.output_dir}  ({size_mb:.0f} Mo)")


if __name__ == "__main__":
    main()
