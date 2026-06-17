from scripts.clean_data import (
    accent_ratio,
    clean_corpus,
    normalize_text,
    symbol_ratio,
)


def test_normalize_typographic_to_ascii():
    assert normalize_text("l’air") == "l'air"
    assert normalize_text("« bonjour »") == '" bonjour "'
    assert normalize_text("attends…") == "attends..."
    assert normalize_text("un—deux") == "un-deux"


def test_normalize_keeps_accents():
    assert normalize_text("éàçùê") == "éàçùê"


def test_normalize_collapses_whitespace():
    assert normalize_text("a   b\t c") == "a b c"


def test_normalize_strips_replacement_char():
    assert "�" not in normalize_text("a�b")


def test_accent_ratio():
    assert accent_ratio("eee") == 0.0
    assert accent_ratio("ééé") == 1.0


def test_symbol_ratio():
    assert symbol_ratio("abcd.") < 0.3
    assert symbol_ratio("@#$%") > 0.5


def test_clean_corpus_filters_and_dedup(tmp_path):
    source = tmp_path / "in.txt"
    source.write_text(
        "\n".join(
            [
                "Le chat dort sur le canapé du salon.",
                "trop court",
                "Une autre phrase bien formée ici.",
                "Le chat dort sur le canapé du salon.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.txt"
    stats = clean_corpus(
        [source], out, min_chars=10, min_accent_ratio=0.0,
        accent_min_len=60, max_symbol_ratio=0.3, dedup=True,
    )
    kept = out.read_text(encoding="utf-8").splitlines()
    assert "Le chat dort sur le canapé du salon." in kept
    assert "Une autre phrase bien formée ici." in kept
    assert "trop court" not in kept
    assert stats["train"] == 2
    assert stats["dup"] == 1


def test_clean_corpus_validation_split(tmp_path):
    source = tmp_path / "in.txt"
    source.write_text(
        "\n".join(f"Phrase numéro {i} bien formée ok." for i in range(100)) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.txt"
    val = tmp_path / "val.txt"
    stats = clean_corpus(
        [source], out, min_chars=10, min_accent_ratio=0.0,
        accent_min_len=60, max_symbol_ratio=0.3, val_output=val, val_every=10,
    )
    assert stats["val"] == 10
    assert stats["train"] == 90
    assert len(val.read_text(encoding="utf-8").splitlines()) == 10
