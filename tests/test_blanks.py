def test_guess_pos():
    from backend.blanks import guess_pos

    assert guess_pos("manger") == "verbe"
    assert guess_pos("heureux") == "adjectif"
    assert guess_pos("maison") == "nom"


def test_punch_holes_structure():
    from backend.blanks import punch_holes

    text = "Le capitaine range son galion dans le frigo immédiatement."
    result = punch_holes(text, n_blanks=2)

    assert len(result["blanks"]) == 2
    assert "{{0}}" in result["template"]
    assert "{{1}}" in result["template"]

    for blank in result["blanks"]:
        assert "[MASK]" in blank["masked_text"]
        assert blank["answer"] in text
        assert blank["hint"] in {"nom", "verbe", "adjectif"}


def test_punch_holes_caps_blanks_to_candidates():
    from backend.blanks import punch_holes

    result = punch_holes("Le chat dort sur le lit.", n_blanks=8)
    assert 1 <= len(result["blanks"]) <= 8


def test_punch_holes_protects_opening():
    from backend.blanks import punch_holes

    opening = "Ilan lit un grand livre."
    text = opening + " Le capitaine range son galion immédiatement."
    result = punch_holes(text, n_blanks=2, protect_until=len(opening))

    chosen = {blank["answer"] for blank in result["blanks"]}
    assert chosen.isdisjoint({"Ilan", "grand", "livre"})
