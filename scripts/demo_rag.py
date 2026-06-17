"""Démo RAG : montre la récupération et l'effet du conditionnement sur la génération.

Pour le rendu : compare, sur les mêmes débuts, la génération du mini-GPT SANS RAG
puis AVEC RAG (passages récupérés en contexte + filet de sécurité), et affiche les
passages récupérés via le mini-BERT.

    python scripts/demo_rag.py
    python scripts/demo_rag.py --opening "Marie trouva une lettre sous le banc."
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.blanks import make_story  # noqa: E402
from backend.generation import make_generator  # noqa: E402
from backend.retrieval import Retriever  # noqa: E402


DEFAULT_OPENINGS = [
    "Le jeune homme entra dans la maison et entendit un bruit derriere la porte.",
    "A minuit, une lumiere apparut soudain au sommet de la vieille tour.",
    "Marie trouva une lettre sans signature sous le banc du jardin.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Démo RAG (récupération + conditionnement).")
    parser.add_argument("--opening", type=str, default=None, help="Un début précis (sinon 3 exemples).")
    parser.add_argument("--lineage", type=str, default="from-scratch", choices=["from-scratch", "pretrained"])
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    openings = [args.opening] if args.opening else DEFAULT_OPENINGS

    print("Chargement du générateur et du retriever...")
    generator = make_generator(args.lineage)
    retriever = Retriever()
    print(f"Index : {retriever.meta['n_passages']:,} passages, encodeur {Path(retriever.meta['encoder_checkpoint']).name}\n")

    for opening in openings:
        print("=" * 78)
        print("DEBUT :", opening)
        print("-" * 78)
        print("Passages recuperes (mini-BERT) :")
        for hit in retriever.search(opening, top_k=args.top_k):
            print(f"  [{hit['score']:.3f}] {hit['text'][:100]}")

        sans = make_story(generator, opening, n_blanks=4, seed=7)
        avec = make_story(generator, opening, n_blanks=4, seed=7, retriever=retriever)
        print("\nGENERATION SANS RAG :")
        print(" ", sans["story"][len(opening):].strip()[:220] or "(vide)")
        print("\nGENERATION AVEC RAG (contexte + filet) :")
        print(" ", avec["story"][len(opening):].strip()[:220] or "(vide)")
        print()


if __name__ == "__main__":
    main()
