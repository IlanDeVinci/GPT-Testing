"""Entraîne un tokenizer BPE Hugging Face à partir du corpus nettoyé."""

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "clean.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def train_tokenizer(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus nettoyé introuvable : {corpus_path}\n"
            "Lancez d'abord : python scripts/clean_data.py"
        )

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne un tokenizer BPE.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Nombre minimal d'occurrences avant une fusion BPE (défaut : 2).",
    )
    parser.add_argument(
        "--test_sentence",
        default="Le jeune homme entra dans la maison avec prudence.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    tokenizer = train_tokenizer(
        args.corpus,
        args.output,
        args.vocab_size,
        args.min_frequency,
    )

    encoded = tokenizer.encode(args.test_sentence)
    print(f"Tokenizer sauvegardé : {args.output}")
    print(f"Taille réelle         : {tokenizer.get_vocab_size()} tokens")
    print(f"Phrase test            : {args.test_sentence}")
    print(f"Tokens                 : {encoded.tokens}")
    print(f"IDs                    : {encoded.ids}")
    print(f"Texte décodé           : {tokenizer.decode(encoded.ids)}")


if __name__ == "__main__":
    main()
