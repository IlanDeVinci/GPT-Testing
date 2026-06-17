"""Construit des paires de préférence (prompt, chosen, rejected) pour le DPO.

Pour chaque prompt, on génère K suites avec le modèle actif (SFT), on les note avec
une récompense transparente, et on garde la meilleure (chosen) vs la pire (rejected).

Récompense (plus haut = mieux) :
- coeur : score du JUGE BERT (plausibilité de "prompt + suite" sous le MLM) ;
- garde-fous : pénalise [UNK], répétitions (3-grammes), fin de phrase manquante ;
- petit bonus : recouvrement de mots-contenu avec le prompt.

À lancer sur GPU, APRÈS le SFT :
    python scripts/build_dpo_data.py --num_prompts 1500 --output data/dpo_pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.blanks import STOPWORDS, WORD_RE  # noqa: E402
from backend.generation import GPTStoryGenerator, make_generator  # noqa: E402
from scripts.bert_reward import BertJudge  # noqa: E402


def content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD_RE.findall(text)
        if len(word) >= 4 and word.lower() not in STOPWORDS
    }


def reward(prompt: str, continuation: str, judge: BertJudge) -> float:
    # Coeur : le juge BERT note la plausibilité de "prompt + suite".
    score = judge.score(f"{prompt} {continuation}")

    # Garde-fous contre les sorties dégénérées (sinon le reward se fait "hacker").
    if "[UNK]" in continuation:
        score -= 2.0
    if not continuation.rstrip().endswith((".", "!", "?")):
        score -= 0.3
    words = continuation.lower().split()
    trigrams = list(zip(words, words[1:], words[2:]))
    if trigrams:
        score -= 1.5 * (1 - len(set(trigrams)) / len(trigrams))

    # Petit bonus explicite : faire référence au prompt.
    prompt_words = content_words(prompt)
    if prompt_words:
        score += 0.5 * len(prompt_words & content_words(continuation)) / len(prompt_words)

    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit des paires DPO.")
    parser.add_argument("--sft_checkpoint", type=Path, default=None,
                        help="Checkpoint à utiliser (défaut : from-scratch via backend/paths).")
    parser.add_argument("--pretrained", action="store_true",
                        help="Générer avec le lignage pré-entraîné (checkpoints-pre-*).")
    parser.add_argument("--prompts_corpus", type=Path, default=PROJECT_ROOT / "data" / "clean-narratif.txt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "dpo_pairs.jsonl")
    parser.add_argument("--num_prompts", type=int, default=1500)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min_gap", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pretrained:
        generator = make_generator("pretrained")
    elif args.sft_checkpoint:
        generator = GPTStoryGenerator(checkpoint=args.sft_checkpoint)
    else:
        generator = GPTStoryGenerator()
    judge = BertJudge()
    print(f"Modèle SFT : {generator.checkpoint.name}")
    print(f"Juge BERT  : {judge.checkpoint.name}")

    prompts = [
        line.strip()
        for line in args.prompts_corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.num_prompts]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts):
            try:
                candidates = generator.generate(
                    prompt,
                    num_candidates=args.k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed=index,
                )
            except RuntimeError:
                continue

            scored = sorted(((reward(prompt, c, judge), c) for c in candidates), reverse=True)
            best_score, best = scored[0]
            worst_score, worst = scored[-1]
            if best == worst or best_score - worst_score < args.min_gap:
                continue

            handle.write(
                json.dumps(
                    {"prompt": prompt, "chosen": best, "rejected": worst},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if index % 200 == 0:
                print(f"  {index} prompts -> {written} paires")

    print(f"Paires écrites : {written} -> {args.output}")


if __name__ == "__main__":
    main()
