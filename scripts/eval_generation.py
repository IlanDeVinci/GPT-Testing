"""Mesure la qualité de génération du mini-GPT : taux de caractères corrompus + échantillons.

A lancer APRES l'entraînement (utilise le GPU). Le backend résout automatiquement le
dernier modèle (checkpoints-v3 si présent).

    python scripts/eval_generation.py
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.generation import GPTStoryGenerator  # noqa: E402


DEFAULT_PROMPTS = [
    "Marie ouvrit la lettre et",
    "Le vieux phare clignota trois fois",
    "À minuit, dans la forêt, un bruit",
    "Le capitaine posa son épée et",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualité de génération du mini-GPT.")
    parser.add_argument("--num_per_prompt", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = GPTStoryGenerator()
    print(f"Checkpoint : {generator.checkpoint.name}\n")

    total_chars = 0
    corrupted = 0
    samples: list[tuple[str, str]] = []
    for prompt in DEFAULT_PROMPTS:
        outputs = generator.generate(
            prompt,
            num_candidates=args.num_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=123,
        )
        for text in outputs:
            total_chars += len(text)
            corrupted += text.count("[UNK]") + text.count("�")
            samples.append((prompt, text))

    rate = corrupted / max(total_chars, 1) * 1000
    print(f"Caractères corrompus / 1000 caractères : {rate:.2f}  (objectif ~0)\n")
    print("Échantillons :")
    for prompt, text in samples:
        print(f"\n  [{prompt}]\n  → {text}")


if __name__ == "__main__":
    main()
