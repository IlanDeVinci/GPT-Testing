"""Chemins des modèles entraînés — autonome, sans dépendance à l'ancien jeu.

Seuls les artefacts entraînés (checkpoints + tokenizers) sont référencés ici.
Le dossier `story_game/` et `app.py` peuvent être supprimés sans impact.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# --- Lignage FROM-SCRATCH (modèle maison) : DÉFAUT du jeu ---
# Priorité : narratif (spécialisé domaine) > v4 (~34M Chinchilla-optimal) > v3.
# L'évaluation a montré que SFT puis DPO dégradent la perplexité du from-scratch
# (365 → 842 → 880 ppl/mot) : on les relègue en dernier. Le modèle spécialisé
# narratif (scripts/train_narratif.ps1) est le meilleur sur le domaine du jeu.
_NARR_READY = (PROJECT_ROOT / "checkpoints-narratif" / "latest.json").exists()
_V4_READY = (PROJECT_ROOT / "checkpoints-v4" / "latest.json").exists()
_DPO_READY = (PROJECT_ROOT / "checkpoints-dpo" / "latest.json").exists()
_SFT_READY = (PROJECT_ROOT / "checkpoints-sft" / "latest.json").exists()
_V3_READY = (PROJECT_ROOT / "checkpoints-v3" / "latest.json").exists()

FROMSCRATCH_DIR = (
    PROJECT_ROOT / "checkpoints-narratif"
    if _NARR_READY
    else PROJECT_ROOT / "checkpoints-v4"
    if _V4_READY
    else PROJECT_ROOT / "checkpoints-v3"
    if _V3_READY
    else PROJECT_ROOT / "checkpoints-sft"
    if _SFT_READY
    else PROJECT_ROOT / "checkpoints-dpo"
    if _DPO_READY
    else PROJECT_ROOT / "checkpoints-story-v2"
    if (PROJECT_ROOT / "checkpoints-story-v2" / "latest.json").exists()
    else PROJECT_ROOT / "checkpoints-trained"
)

# --- Lignage PRÉ-ENTRAÎNÉ (modèle HF fine-tuné) : option ---
# Son tokenizer est sauvegardé DANS le checkpoint → chargé via AutoTokenizer.
# pre-sft préféré à pre-dpo : le DPO a dégradé la perplexité (355 → 374).
_PRE_DPO = (PROJECT_ROOT / "checkpoints-pre-dpo" / "latest.json").exists()
_PRE_SFT = (PROJECT_ROOT / "checkpoints-pre-sft" / "latest.json").exists()
PRETRAINED_DIR = (
    PROJECT_ROOT / "checkpoints-pre-sft"
    if _PRE_SFT
    else PROJECT_ROOT / "checkpoints-pre-dpo"
    if _PRE_DPO
    else None
)

GPT_TOKENIZER_PATH = (
    PROJECT_ROOT / "tokenizer-v3" / "tokenizer.json"
    if (_DPO_READY or _SFT_READY or _V3_READY)
    else PROJECT_ROOT / "tokenizer-trained" / "tokenizer.json"
    if (PROJECT_ROOT / "tokenizer-trained" / "tokenizer.json").exists()
    else PROJECT_ROOT / "tokenizer" / "tokenizer.json"
)

# Défaut = from-scratch ("le mien"). Le jeu peut basculer vers le pré-entraîné.
GPT_CHECKPOINTS_DIR = FROMSCRATCH_DIR
GPT_IS_PRETRAINED = False


def available_gpt_lineages() -> list[dict]:
    """Lignages GPT proposés au jeu (id, libellé, disponibilité)."""
    return [
        {
            "id": "from-scratch",
            "label": "Mon modèle (from scratch)",
            "available": (FROMSCRATCH_DIR / "latest.json").exists(),
        },
        {
            "id": "pretrained",
            "label": "Pré-entraîné (français)",
            "available": PRETRAINED_DIR is not None,
        },
    ]


def resolve_gpt_lineage(lineage: str) -> dict:
    """Décrit un lignage GPT et résout son checkpoint local si disponible."""
    lineages = {item["id"]: item for item in available_gpt_lineages()}
    if lineage not in lineages:
        raise KeyError(f"Lignage GPT inconnu : {lineage}")

    item = dict(lineages[lineage])
    root = FROMSCRATCH_DIR if lineage == "from-scratch" else PRETRAINED_DIR
    item["checkpoint"] = None
    item["checkpoint_dir"] = None
    item["is_pretrained"] = lineage == "pretrained"
    if root is not None and item["available"]:
        checkpoint = find_latest_checkpoint(root)
        item["checkpoint"] = checkpoint.name
        item["checkpoint_dir"] = str(checkpoint)
    return item

# Préférer le BERT réentraîné (mlm-v3 / tokenizer-v3) s'il existe, sinon l'ancien.
BERT_TOKENIZER_DIR = (
    PROJECT_ROOT / "tinybert" / "tokenizer-v3"
    if (PROJECT_ROOT / "tinybert" / "tokenizer-v3" / "tokenizer.json").exists()
    else PROJECT_ROOT / "tinybert" / "tokenizer-gutenberg-16k"
)
BERT_MLM_DIR = (
    PROJECT_ROOT / "tinybert" / "mlm-v3"
    if (PROJECT_ROOT / "tinybert" / "mlm-v3" / "latest.json").exists()
    else PROJECT_ROOT / "tinybert" / "mlm-gutenberg-v3"
)


def find_latest_checkpoint(root: Path) -> Path:
    """Trouve le checkpoint indiqué par latest.json ou le plus récent."""
    latest_file = root / "latest.json"
    if latest_file.exists():
        data = json.loads(latest_file.read_text(encoding="utf-8"))
        checkpoint = root / data["checkpoint"]
        if checkpoint.exists():
            return checkpoint

    candidates = [
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir() and (path / "config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"Aucun checkpoint trouvé dans {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)
