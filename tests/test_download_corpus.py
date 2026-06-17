from scripts.download_french_corpus import iter_lines


def test_iter_lines_splits_sentences():
    text = "Phrase une ici. Phrase deux ici! Et trois alors?"
    out = list(iter_lines(text, split_sentences=True, min_chars=5, max_chars=100))
    assert "Phrase une ici." in out
    assert "Phrase deux ici!" in out
    assert "Et trois alors?" in out


def test_iter_lines_no_split_returns_whole_line():
    out = list(iter_lines("Une ligne entière.", split_sentences=False, min_chars=5, max_chars=100))
    assert out == ["Une ligne entière."]


def test_iter_lines_length_filter():
    assert list(iter_lines("court", split_sentences=False, min_chars=10, max_chars=100)) == []
    assert list(iter_lines("x" * 500, split_sentences=False, min_chars=10, max_chars=100)) == []
