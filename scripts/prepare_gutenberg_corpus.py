"""Prépare les 218 livres français de Project Gutenberg par ouvrage."""

import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    PROJECT_ROOT / "data" / "gutenberg-source" / "gutemberg-txt-fr.zip"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "gutenberg"

START_RE = re.compile(r"\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG", re.I)
END_RE = re.compile(r"\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG", re.I)
SPACE_RE = re.compile(r"[ \t]+")
FINAL_PUNCTUATION_RE = re.compile(r"[.!?…][\"'»)\]]*$")


def decode_book(data: bytes) -> tuple[str, str]:
    """Décode les encodages Gutenberg courants sans perdre les accents."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", data, 0, 1, "encodage non reconnu")


def metadata_value(lines: list[str], field: str) -> str:
    prefix = f"{field.lower()}:"
    for line in lines[:100]:
        if line.strip().lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def strip_gutenberg_boilerplate(lines: list[str]) -> list[str]:
    start = next(
        (index + 1 for index, line in enumerate(lines) if START_RE.search(line)),
        0,
    )
    end = next(
        (
            index
            for index in range(len(lines) - 1, start - 1, -1)
            if END_RE.search(lines[index])
        ),
        len(lines),
    )
    return lines[start:end]


def build_paragraphs(
    lines: list[str],
    min_chars: int,
    max_chars: int,
) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = SPACE_RE.sub(" ", " ".join(buffer)).strip()
        buffer.clear()
        if len(text) < min_chars:
            return
        # Les très longs paragraphes sont découpés sur les fins de phrase.
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        chunk: list[str] = []
        chunk_length = 0
        for sentence in sentences:
            if chunk and chunk_length + len(sentence) + 1 > max_chars:
                candidate = " ".join(chunk).strip()
                if FINAL_PUNCTUATION_RE.search(candidate):
                    paragraphs.append(candidate)
                chunk = []
                chunk_length = 0
            chunk.append(sentence)
            chunk_length += len(sentence) + 1
        candidate = " ".join(chunk).strip()
        if len(candidate) >= min_chars and FINAL_PUNCTUATION_RE.search(candidate):
            paragraphs.append(candidate)

    for raw_line in lines:
        line = SPACE_RE.sub(" ", raw_line.replace("\u00a0", " ")).strip()
        if not line:
            flush()
        else:
            buffer.append(line)
    flush()
    return paragraphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nettoie et sépare les livres Gutenberg français."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=700)
    parser.add_argument(
        "--min_accent_rate",
        type=float,
        default=0.0,
        help="Filtre les livres ASCII; 0.001 convient à un GPT français accentué.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(
            f"Archive absente : {args.archive}\n"
            "Téléchargez cabusar/gutenberg-txt-fr depuis Hugging Face."
        )

    books: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            text, encoding = decode_book(archive.read(info))
            original_lines = text.splitlines()
            paragraphs = build_paragraphs(
                strip_gutenberg_boilerplate(original_lines),
                args.min_chars,
                args.max_chars,
            )
            if len(paragraphs) < 20:
                continue
            joined = " ".join(paragraphs)
            letter_count = sum(character.isalpha() for character in joined)
            accent_count = sum(
                character in "àâäçéèêëîïôöùûüÿœæÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸŒÆ"
                for character in joined
            )
            accent_rate = accent_count / max(letter_count, 1)
            if accent_rate < args.min_accent_rate:
                continue
            books.append(
                {
                    "book_id": Path(info.filename).stem,
                    "title": metadata_value(original_lines, "Title"),
                    "author": metadata_value(original_lines, "Author"),
                    "encoding": encoding,
                    "accent_rate": accent_rate,
                    "paragraphs": paragraphs,
                }
            )

    if len(books) < 100:
        raise RuntimeError(f"Seulement {len(books)} livres exploitables trouvés.")

    rng = random.Random(args.seed)
    rng.shuffle(books)
    validation_count = max(1, round(len(books) * args.validation_ratio))
    validation_books = books[:validation_count]
    train_books = books[validation_count:]
    for book in train_books:
        book["split"] = "train"
    for book in validation_books:
        book["split"] = "validation"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "books.jsonl").open("w", encoding="utf-8") as output:
        for book in books:
            output.write(json.dumps(book, ensure_ascii=False) + "\n")

    def write_split(name: str, selected: list[dict[str, object]]) -> int:
        paragraphs = [
            paragraph
            for book in selected
            for paragraph in book["paragraphs"]  # type: ignore[index]
        ]
        (args.output_dir / f"{name}.txt").write_text(
            "\n".join(paragraphs) + "\n",
            encoding="utf-8",
        )
        return len(paragraphs)

    train_paragraphs = write_split("train", train_books)
    validation_paragraphs = write_split("validation", validation_books)
    (args.output_dir / "clean.txt").write_text(
        (args.output_dir / "train.txt").read_text(encoding="utf-8")
        + (args.output_dir / "validation.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    summary = {
        "source_dataset": "cabusar/gutenberg-txt-fr",
        "source_url": "https://huggingface.co/datasets/cabusar/gutenberg-txt-fr",
        "upstream_source": "Project Gutenberg",
        "upstream_url": "https://www.gutenberg.org/",
        "archive": str(args.archive),
        "books_total": len(books),
        "books_train": len(train_books),
        "books_validation": len(validation_books),
        "paragraphs_train": train_paragraphs,
        "paragraphs_validation": validation_paragraphs,
        "split_unit": "book",
        "seed": args.seed,
        "min_accent_rate": args.min_accent_rate,
    }
    (args.output_dir / "corpus_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
