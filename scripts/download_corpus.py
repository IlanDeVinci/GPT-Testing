"""Télécharge un corpus narratif français public depuis OPUS Books."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw.txt"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "corpus_info.json"
SPACE_RE = re.compile(r"\s+")
FINAL_PUNCTUATION_RE = re.compile(r"[.!?…][\"'»)\]]*$")


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Télécharge OPUS Books en français.")
    parser.add_argument("--dataset", default="Helsinki-NLP/opus_books")
    parser.add_argument("--config", default="en-fr")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--max_lines", type=int, default=25000)
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--max_chars", type=int, default=600)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lit le dataset en streaming sans télécharger toute la collection.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.max_lines < 1000:
        raise ValueError("--max_lines doit être au moins égal à 1000.")

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
    )

    selected: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for example in dataset:
        scanned += 1
        translation = example.get("translation", {})
        text = normalize(str(translation.get("fr", "")))
        if not (args.min_chars <= len(text) <= args.max_chars):
            continue
        if not FINAL_PUNCTUATION_RE.search(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(text)
        if len(selected) >= args.max_lines:
            break

    if len(selected) < 1000:
        raise RuntimeError(
            f"Seulement {len(selected)} lignes valides trouvées après {scanned} exemples."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    metadata = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "source_url": f"https://huggingface.co/datasets/{args.dataset}",
        "upstream_project": "https://opus.nlpl.eu/",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "streaming": args.streaming,
        "examples_scanned": scanned,
        "lines_selected": len(selected),
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "usage_note": (
            "OPUS Books contient des livres libres de droits. La fiche du dataset "
            "indique un usage personnel, éducatif et de recherche."
        ),
    }
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Corpus écrit dans : {args.output}")


if __name__ == "__main__":
    main()

