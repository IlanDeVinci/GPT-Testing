"""Entraîne un tokenizer WordPiece pour le mini-BERT."""

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFC, Sequence
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.paths import BERT_TOKENIZER_DIR, DATA_DIR  # noqa: E402


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_wordpiece(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
) -> PreTrainedTokenizerFast:
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {corpus_path}. Lancez clean_data.py."
        )

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFC(), Lowercase()])
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = WordPieceDecoder(prefix="##")

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=256,
    )
    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne le WordPiece du mini-BERT.")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "clean.txt")
    parser.add_argument("--output_dir", type=Path, default=BERT_TOKENIZER_DIR)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument(
        "--test_sentence",
        default="Le jeune homme entra dans la maison avec prudence.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    tokenizer = train_wordpiece(
        args.corpus,
        args.output_dir,
        args.vocab_size,
        args.min_frequency,
    )
    encoded = tokenizer(args.test_sentence)
    print(f"Tokenizer sauvegardé : {args.output_dir}")
    print(f"Taille réelle         : {len(tokenizer)}")
    print(f"Tokens                 : {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
    print(f"IDs                    : {encoded['input_ids']}")


if __name__ == "__main__":
    main()

