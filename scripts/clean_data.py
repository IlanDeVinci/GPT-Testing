"""Nettoie et normalise un ou plusieurs corpus texte ligne par ligne.

Objectif qualité : un tokenizer ByteLevel encode chaque caractere en octets
UTF-8. Les caracteres multi-octets peu frequents (apostrophe courbe ', guillemets
« », tiret cadratin —) sont fragiles : un modele sous-entraine emet une sequence
d'octets invalide et le decodage produit un caractere de remplacement. On normalise
donc la ponctuation typographique vers l'ASCII (1 octet) tout en GARDANT les accents
francais, qui sont frequents et reellement appris.
"""

import argparse
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [PROJECT_ROOT / "data" / "raw.txt"]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "clean.txt"

WHITESPACE_RE = re.compile(r"\s+")
FINAL_PUNCTUATION_RE = re.compile(r"[.!?][\"')\]]*$")

TYPOGRAPHIC_MAP = {
    "’": "'", "‘": "'", "‚": "'", "‛": "'", "ʼ": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
    "…": "...",
    "–": "-", "—": "-", "―": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
}
TYPOGRAPHIC_RE = re.compile("|".join(map(re.escape, TYPOGRAPHIC_MAP)))

FRENCH_ACCENTS = set("àâäçéèêëîïôöùûüÿœæ")
ALLOWED_PUNCTUATION = set(".,;:!?'\"-()")

# Restes de balisage wiki/HTML (surtout wikisource) : jamais du français propre.
MARKUP_RE = re.compile(r"[|*_#=<>\[\]{}~]|style=|rowspan|colspan|http")


def normalize_text(line: str) -> str:
    """NFC + ponctuation typographique vers ASCII, sans toucher aux accents."""
    line = unicodedata.normalize("NFC", line)
    line = line.replace("�", "")
    line = TYPOGRAPHIC_RE.sub(lambda match: TYPOGRAPHIC_MAP[match.group()], line)
    return WHITESPACE_RE.sub(" ", line).strip()


def accent_ratio(line: str) -> float:
    """Fraction de lettres accentuees — sert a rejeter le faux francais sans accents."""
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return 1.0
    accented = sum(1 for char in letters if char.lower() in FRENCH_ACCENTS)
    return accented / len(letters)


def symbol_ratio(line: str) -> float:
    """Fraction de caracteres ni alphanumeriques ni ponctuation courante."""
    visible = [char for char in line if not char.isspace()]
    if not visible:
        return 1.0
    weird = sum(
        1 for char in visible if not (char.isalnum() or char in ALLOWED_PUNCTUATION)
    )
    return weird / len(visible)


def clean_corpus(
    input_paths: list[Path],
    output_path: Path,
    min_chars: int,
    min_accent_ratio: float,
    accent_min_len: int,
    max_symbol_ratio: float,
    val_output: Path | None = None,
    val_every: int = 50,
    dedup: bool = True,
    reject_markup: bool = False,
) -> dict[str, int]:
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Corpus introuvable : {input_path}")

    seen: set[str] = set()
    stats = {
        "read": 0, "short": 0, "no_final": 0, "symbols": 0,
        "accents": 0, "markup": 0, "dup": 0, "train": 0, "val": 0,
    }
    kept_total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_handle = output_path.open("w", encoding="utf-8")
    val_handle = val_output.open("w", encoding="utf-8") if val_output else None
    try:
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8", errors="replace") as source:
                for raw_line in source:
                    stats["read"] += 1
                    if stats["read"] % 2_000_000 == 0:
                        print(
                            f"  {stats['read']:,} lues -> {stats['train']:,} conservees",
                            flush=True,
                        )
                    line = normalize_text(raw_line)

                    if len(line) < min_chars:
                        stats["short"] += 1
                        continue
                    if not FINAL_PUNCTUATION_RE.search(line):
                        stats["no_final"] += 1
                        continue
                    if symbol_ratio(line) > max_symbol_ratio:
                        stats["symbols"] += 1
                        continue
                    if reject_markup and MARKUP_RE.search(line):
                        stats["markup"] += 1
                        continue
                    if len(line) >= accent_min_len and accent_ratio(line) < min_accent_ratio:
                        stats["accents"] += 1
                        continue
                    if dedup:
                        if line in seen:
                            stats["dup"] += 1
                            continue
                        seen.add(line)
                    kept_total += 1
                    # 1 ligne conservee sur val_every va dans la validation (held-out).
                    if val_handle is not None and kept_total % val_every == 0:
                        val_handle.write(line + "\n")
                        stats["val"] += 1
                    else:
                        train_handle.write(line + "\n")
                        stats["train"] += 1
    finally:
        train_handle.close()
        if val_handle is not None:
            val_handle.close()

    if stats.get("train", 0) == 0:
        raise ValueError("Aucune ligne conservee. Assouplissez les filtres.")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nettoie et normalise le corpus francais.")
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument(
        "--min_accent_ratio",
        type=float,
        default=0.0,
        help="Rejette les lignes longues sous ce taux d'accents (0 = desactive). "
        "Conseil pour fusionner du Gutenberg ancien : 0.012.",
    )
    parser.add_argument(
        "--accent_min_len",
        type=int,
        default=60,
        help="Le filtre d'accents ne s'applique qu'aux lignes d'au moins N caracteres.",
    )
    parser.add_argument("--max_symbol_ratio", type=float, default=0.08)
    parser.add_argument(
        "--val_output",
        type=Path,
        default=None,
        help="Si fourni, ecrit un set de validation tenu a l'ecart dans ce fichier.",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=50,
        help="1 ligne conservee sur N va dans la validation (defaut 50 = 2%%).",
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplique en RAM. A desactiver (--no-dedup) sur les tres gros corpus.",
    )
    parser.add_argument(
        "--reject_markup",
        action="store_true",
        help="Rejette les lignes contenant du balisage wiki/HTML (| * = [] style= ...). "
        "Recommande pour wikisource.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = clean_corpus(
        args.inputs,
        args.output,
        args.min_chars,
        args.min_accent_ratio,
        args.accent_min_len,
        args.max_symbol_ratio,
        args.val_output,
        args.val_every,
        args.dedup,
        args.reject_markup,
    )
    print(f"Lignes lues         : {stats['read']}")
    print(f"  rejet trop court  : {stats['short']}")
    print(f"  rejet ponctuation : {stats['no_final']}")
    print(f"  rejet symboles    : {stats['symbols']}")
    print(f"  rejet sans accents: {stats['accents']}")
    print(f"  rejet markup      : {stats['markup']}")
    print(f"  doublons          : {stats['dup']}")
    print(f"Entrainement        : {stats['train']} lignes -> {args.output}")
    if args.val_output:
        print(f"Validation          : {stats['val']} lignes -> {args.val_output}")


if __name__ == "__main__":
    main()
