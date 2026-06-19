"""API du jeu Histoires Trouées : le mini-GPT invente, le mini-BERT remplit.

Lancer depuis la racine du projet :

    uvicorn backend.main:app --reload
"""

import random
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.blanks import make_story
from backend.generation import GPTStoryGenerator, make_generator
from backend.mask_fill import MaskFiller
from backend.paths import available_gpt_lineages, resolve_gpt_lineage
from backend.retrieval import Retriever


OPENINGS = [
    "Le jeune homme entra dans la maison et entendit un bruit derrière la porte.",
    "À minuit, une lumière apparut soudain au sommet de la vieille tour.",
    "Marie trouva une lettre sans signature sous le banc du jardin.",
]

app = FastAPI(title="Histoires Trouées")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_generators: dict[str, GPTStoryGenerator] = {}
# Défaut = from-scratch ("mon modèle"). paths.py résout désormais vers le meilleur
# checkpoint disponible (narratif > v4 > v3), donc plus jamais le DPO dégradé.
_active_lineage = "from-scratch"
_filler: MaskFiller | None = None
_retriever: Retriever | None = None

# Verrous de chargement : le frontend déclenche plusieurs requêtes en parallèle
# (ex. plusieurs /fill d'un coup). Sans verrou, deux threads construisent le même
# modèle en même temps et le chargement concurrent du checkpoint casse
# ("Cannot copy out of meta tensor"). On sérialise donc la première construction.
_generator_lock = threading.Lock()
_filler_lock = threading.Lock()
_retriever_lock = threading.Lock()


def get_generator() -> GPTStoryGenerator:
    if _active_lineage not in _generators:
        with _generator_lock:
            if _active_lineage not in _generators:
                _generators[_active_lineage] = make_generator(_active_lineage)
    return _generators[_active_lineage]


def get_filler() -> MaskFiller:
    global _filler
    if _filler is None:
        with _filler_lock:
            if _filler is None:
                _filler = MaskFiller()
    return _filler


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


def active_model_state() -> dict:
    """État lisible par le frontend, sans forcer le chargement du modèle."""
    state = resolve_gpt_lineage(_active_lineage)
    if _active_lineage in _generators:
        state["checkpoint"] = _generators[_active_lineage].checkpoint.name
        state["loaded"] = True
    else:
        state["loaded"] = False
    return state


class GenerateRequest(BaseModel):
    opening: str | None = None
    n_blanks: int = 4
    seed: int | None = None
    use_rag: bool = False


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class FillRequest(BaseModel):
    masked_text: str
    top_k: int = 5


class ModelRequest(BaseModel):
    lineage: str


@app.get("/models")
def models() -> dict:
    active = active_model_state()
    return {
        "models": available_gpt_lineages(),
        "active": _active_lineage,
        "active_label": active["label"],
        "active_checkpoint": active["checkpoint"],
        "active_loaded": active["loaded"],
    }


@app.post("/model")
def set_model(request: ModelRequest) -> dict:
    global _active_lineage
    lineages = {item["id"]: item for item in available_gpt_lineages()}
    if request.lineage not in lineages:
        raise HTTPException(status_code=422, detail="Lignage de modèle inconnu.")
    if not lineages[request.lineage]["available"]:
        raise HTTPException(
            status_code=422,
            detail="Ce modèle n'est pas disponible (pas encore entraîné).",
        )
    _active_lineage = request.lineage
    generator = get_generator()  # charge maintenant (peut prendre ~20 s)
    return {
        "active": _active_lineage,
        "active_label": lineages[_active_lineage]["label"],
        "active_checkpoint": generator.checkpoint.name,
    }


@app.get("/health")
def health() -> dict:
    return {
        "gpt_loaded": _active_lineage in _generators,
        "active_model": active_model_state(),
        "bert_loaded": _filler is not None,
    }


@app.get("/openings")
def openings() -> dict:
    return {"openings": OPENINGS}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    opening = (request.opening or random.choice(OPENINGS)).strip()
    if not opening:
        raise HTTPException(status_code=422, detail="Le début ne peut pas être vide.")
    seed = request.seed if request.seed is not None else random.randint(1, 10_000)
    retriever = None
    if request.use_rag:
        try:
            retriever = get_retriever()
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
    try:
        return make_story(
            get_generator(),
            opening,
            n_blanks=max(1, min(request.n_blanks, 8)),
            seed=seed,
            retriever=retriever,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="La requête ne peut pas être vide.")
    try:
        retriever = get_retriever()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    hits = retriever.search(request.query, top_k=max(1, min(request.top_k, 10)))
    return {"query": request.query, "results": hits, "index": retriever.meta}


@app.post("/fill")
def fill(request: FillRequest) -> dict:
    if "[MASK]" not in request.masked_text:
        raise HTTPException(status_code=422, detail="Le texte doit contenir [MASK].")
    candidates = get_filler().suggest(
        request.masked_text,
        top_k=max(1, min(request.top_k, 10)),
    )
    return {"candidates": candidates}
