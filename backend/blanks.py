"""Génère une histoire avec le mini-GPT puis y perce des trous à remplir."""

import re

from backend.generation import GPTStoryGenerator


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

STOPWORDS = {
    "alors", "après", "aussi", "autre", "avait", "avec", "avoir", "bien",
    "cela", "cette", "comme", "dans", "depuis", "deux", "donc", "elle",
    "elles", "encore", "entre", "était", "étaient", "êtes", "faire", "fait",
    "leur", "leurs", "lui", "mais", "même", "moins", "notre", "nous", "pour",
    "puis", "quand", "quel", "quelle", "rien", "sans", "ses", "son", "sont",
    "sous", "sur", "tous", "tout", "toute", "très", "une", "vers", "votre",
    "vous", "plus", "celui", "cela", "dont", "où", "que", "qui", "quoi",
}

VERB_SUFFIXES = ("er", "ir", "ait", "aient", "ais", "ent", "ée", "és", "ées")
ADJ_SUFFIXES = ("eux", "euse", "ique", "able", "ible", "if", "ive", "al", "ant")


def guess_pos(word: str) -> str:
    """Devine une catégorie grammaticale par suffixe — heuristique, sans modèle."""
    low = word.lower()
    if low.endswith(ADJ_SUFFIXES):
        return "adjectif"
    if low.endswith(VERB_SUFFIXES):
        return "verbe"
    return "nom"


def punch_holes(text: str, n_blanks: int = 4, protect_until: int = 0) -> dict:
    """Sélectionne des mots-contenu espacés et renvoie le gabarit + les trous.

    Les mots commençant avant ``protect_until`` (ex. le début saisi par le joueur)
    ne sont jamais transformés en trous.
    """
    candidates = [
        (match.start(), match.end(), match.group())
        for match in WORD_RE.finditer(text)
        if match.start() >= protect_until
        and match.group().lower() not in STOPWORDS
        and len(match.group()) >= 4
    ]
    n_blanks = min(n_blanks, len(candidates))
    if n_blanks == 0:
        raise RuntimeError("Histoire trop courte pour y percer des trous.")

    step = len(candidates) / n_blanks
    chosen = sorted(
        (candidates[int(i * step)] for i in range(n_blanks)),
        key=lambda span: span[0],
    )

    parts: list[str] = []
    cursor = 0
    for index, (start, end, _word) in enumerate(chosen):
        parts.append(text[cursor:start])
        parts.append("{{%d}}" % index)
        cursor = end
    parts.append(text[cursor:])
    template = "".join(parts)

    blanks = [
        {
            "index": index,
            "answer": word,
            "hint": guess_pos(word),
            "masked_text": text[:start] + "[MASK]" + text[end:],
        }
        for index, (start, end, word) in enumerate(chosen)
    ]
    return {"template": template, "blanks": blanks}


def make_story(
    generator: GPTStoryGenerator,
    opening: str,
    n_blanks: int = 4,
    max_new_tokens: int = 70,
    temperature: float = 0.7,
    top_p: float = 0.85,
    repetition_penalty: float = 1.18,
    min_chars: int = 80,
    seed: int = 42,
    retriever: object | None = None,
    n_context: int = 2,
    style_context: bool = False,
) -> dict:
    """Génère une continuation au mini-GPT puis perce des trous dans l'histoire.

    Si ``retriever`` est fourni (RAG), il sert TOUJOURS de filet de sécurité (une
    vraie phrase humaine remplace une génération ratée). Le conditionnement de
    style (passages en contexte du GPT) est désactivé par défaut : il aide un
    modèle faible à trouver un thème, mais distrait un modèle déjà fluide. On
    l'active explicitement avec ``style_context=True``.
    """
    opening = opening.strip()
    retrieved: list[dict] = []
    style_prefix = ""
    if retriever is not None:
        retrieved = retriever.search(opening, top_k=max(n_context, 3))
        if style_context:
            # Passages courts en exemple, séparés par des sauts de ligne (comme à
            # l'entraînement continu). La troncature gauche préserve le début.
            examples = [hit["text"][:160] for hit in retrieved[:n_context]]
            if examples:
                style_prefix = "\n".join(examples) + "\n"

    try:
        continuation = generator.generate(
            style_prefix + opening,
            num_candidates=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_chars=min_chars,
            seed=seed,
        )[0]
    except RuntimeError:
        continuation = ""

    # Filet de sécurité RAG : si le modèle n'a rien produit d'exploitable,
    # on utilise une vraie continuation humaine récupérée.
    if (not continuation or len(continuation) < min_chars) and retrieved:
        continuation = retrieved[0]["text"]

    text = f"{opening} {continuation}".strip()
    # On ne perce des trous que dans la suite, jamais dans le début du joueur.
    result = punch_holes(text, n_blanks, protect_until=len(opening))
    result["opening"] = opening
    result["story"] = text
    if retrieved:
        result["retrieved"] = retrieved[:n_context]
    return result
