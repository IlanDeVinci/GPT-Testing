"""Telecharge du francais propre depuis plusieurs sources Hugging Face (streaming).

Sources disponibles (toutes parquet natives, streaming) :
- wikipedia : encyclopedique, bien accentue.
- fineweb   : web filtre/deduplique (HuggingFaceFW/fineweb-2), gros volume et varie.
- wikisource: textes litteraires du domaine public (narratif).

Telecharge chaque source dans son propre fichier, puis fusionne-les avec
scripts/clean_data.py (option --inputs) pour normaliser et filtrer.
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

SOURCES = {
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "config": "20231101.fr",
        "extract": lambda example: example.get("text", ""),
        "split_sentences": True,
    },
    "fineweb": {
        "path": "HuggingFaceFW/fineweb-2",
        "config": "fra_Latn",
        "extract": lambda example: example.get("text", ""),
        "split_sentences": True,
    },
    "wikisource": {
        "path": "wikimedia/wikisource",
        "config": "20231201.fr",
        "extract": lambda example: example.get("text", ""),
        "split_sentences": True,
    },
}


def iter_lines(text: str, split_sentences: bool, min_chars: int, max_chars: int):
    if split_sentences:
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for sentence in SENTENCE_SPLIT_RE.split(paragraph):
                sentence = sentence.strip()
                if min_chars <= len(sentence) <= max_chars:
                    yield sentence
    else:
        line = text.strip()
        if min_chars <= len(line) <= max_chars:
            yield line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telecharge un corpus francais propre.")
    parser.add_argument("--source", choices=sorted(SOURCES), default="wikipedia")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--max_docs",
        type=int,
        default=20000,
        help="Nombre de documents a parcourir (articles / pages / documents web).",
    )
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--max_chars", type=int, default=400)
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplique les lignes en RAM. A desactiver (--no-dedup) sur les gros "
        "corpus deja dedupliques comme fineweb, pour eviter de saturer la memoire.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = SOURCES[args.source]
    output = args.output or PROJECT_ROOT / "data" / f"{args.source}-fr.txt"

    dataset = load_dataset(
        source["path"], source["config"], split="train", streaming=True
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    docs = 0
    written = 0

    with output.open("w", encoding="utf-8") as handle:
        for example in dataset:
            text = source["extract"](example)
            for line in iter_lines(
                text, source["split_sentences"], args.min_chars, args.max_chars
            ):
                if args.dedup:
                    if line in seen:
                        continue
                    seen.add(line)
                handle.write(line + "\n")
                written += 1
            docs += 1
            if docs % 5000 == 0:
                print(f"  {docs} exemples -> {written} lignes")
            if docs >= args.max_docs:
                break

    print(f"Source             : {args.source}")
    print(f"Exemples parcourus : {docs}")
    print(f"Lignes ecrites     : {written}")
    print(f"Corpus brut        : {output}")
    print("Etape suivante : fusionner et normaliser avec clean_data.py --inputs.")


if __name__ == "__main__":
    main()
