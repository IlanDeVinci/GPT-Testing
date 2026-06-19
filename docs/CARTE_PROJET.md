# Carte du projet — où trouver quoi

## Le code (petit, versionné par Git)

| Dossier / fichier | Contenu |
|---|---|
| `backend/` | L'API du jeu (FastAPI) |
| `backend/main.py` | Routes `/generate`, `/fill`, `/model`, `/retrieve`, `/openings` |
| `backend/generation.py` | Chargement + inférence du mini-GPT |
| `backend/mask_fill.py` | Remplissage de trous par le mini-BERT |
| `backend/blanks.py` | Génère l'histoire + perce les trous (+ intégration RAG) |
| `backend/retrieval.py` | RAG : recherche sémantique via le mini-BERT |
| `backend/paths.py` | Résout quel checkpoint utiliser (narratif > v4 > v3 …) |
| `frontend/` | Interface React + Vite (`src/`) |
| `scripts/` | Tous les scripts d'entraînement / éval / outils (voir plus bas) |
| `tests/` | Tests pytest |
| `docs/` | Toute la doc (rendu, historique, migration, cette carte) |
| `*.ps1` (racine) | `start.ps1`, `stop.ps1` : lancer / arrêter le jeu |

## Les données — dossier `data/` (~7,5 Go, NON versionné)

| Fichier | Rôle |
|---|---|
| `clean-v3.txt` (2,4 Go) | **Corpus de pré-entraînement** du GPT from-scratch |
| `clean-v3-val.txt` (50 Mo) | Validation perplexité (français général) |
| `narratif-train.txt` (53 Mo) | **Corpus de spécialisation narrative** |
| `narratif-val.txt` (2 Mo) | **Validation narrative** (le test qui compte pour le jeu) |
| `clean-bert.txt` (215 Mo) | Corpus du mini-BERT (MLM) |
| `rag-index/` (102 Mo) | Index RAG (embeddings + passages) |
| ~~`fineweb-fr.txt`, `wikipedia-fr.txt`, `raw.txt`~~ | Sources brutes **supprimées** (libéré ~3 Go ; `clean-v3` déjà construit). Re-téléchargeables via `scripts/download_*.py` si besoin |
| `*.tokens.bin` / `.meta.json` | Caches de tokenisation (régénérés automatiquement) |
| `dpo_pairs*.jsonl` | Paires de préférences (DPO) |

## Les entraînements — dossiers `checkpoints-*` et `tinybert/`

Chaque dossier contient un ou plusieurs `checkpoint-...` + un `latest.json` qui
pointe vers le meilleur. Le jeu charge automatiquement via `paths.py`.

| Dossier | Modèle | Params |
|---|---|---|
| **`checkpoints-narratif/`** | **From-scratch SPÉCIALISÉ narratif (le meilleur sur le domaine)** | 33,7 M |
| `checkpoints-v4/` | From-scratch base (~34 M) — si l'entraînement de nuit a abouti | 33,7 M |
| `checkpoints-v3/` | From-scratch base précédente | 97,7 M |
| `checkpoints-sft/`, `checkpoints-dpo/` | Anciennes variantes alignées (dégradées — gardées pour comparaison) | 97,7 M |
| `checkpoints-pre-sft/`, `checkpoints-pre-dpo/` | Lignage pré-entraîné FR (`asi/gpt-fr-cased-small` fine-tuné) | 124,2 M |
| `checkpoints-story-v2/` | Tout premier modèle (obsolète) | — |
| `tinybert/mlm-v3/` | **mini-BERT MLM** (remplissage + encodeur RAG) | 9,0 M |
| `tokenizer-v3/`, `tinybert/tokenizer-v3/` | Tokenizers entraînés (GPT / BERT) | — |

## Les scripts — dossier `scripts/`

| Script | Rôle |
|---|---|
| `train_all.ps1` | **Pipeline complet** : base → spécialisation → index RAG → éval (graphes en direct) |
| `train_tonight.ps1` | Pré-entraînement base from-scratch seul (`checkpoints-v4`) |
| `train_narratif.ps1` | Spécialisation narrative seule (`checkpoints-narratif`) |
| `train_model.py` | Entraîne / continue un GPT (cœur de l'entraînement) |
| `train_tinybert_mlm.py` | Entraîne le mini-BERT |
| `train_tokenizer.py`, `train_bert_tokenizer.py` | Entraînent les tokenizers |
| `clean_data.py` | Nettoie / normalise un corpus (+ split validation) |
| `build_rag_index.py` | Construit l'index RAG |
| `eval_all.py` | **Compare tous les modèles** (perplexité/mot + accuracy + params) |
| `plot_training.py` | Trace les courbes loss/accuracy (live ou PNG) |
| `demo_rag.py` | Démo de la récupération RAG |
| `package_for_transfer.ps1` | Prépare le transfert vers un autre PC |
| `build_dpo_data.py`, `train_sft.py`, `train_dpo.py` | Pipeline SFT/DPO (historique) |

## Les sorties — dossier `outputs/`

Graphes et résultats : `curve-*.png` (courbes d'entraînement), `eval_all.png` /
`eval_narratif.png` (comparatifs), `eval_*.json` (chiffres bruts).

## Le reste

| Dossier | Contenu |
|---|---|
| `archives/` | Anciens artefacts mis de côté (À CONSERVER) |
| `backups/` | Sauvegardes zip (ex. `from-scratch-v3.zip`) |
| `md/` | `slides.pptx` (présentation) |
| `logs/` | Journaux d'exécution |
| `.venv-gpu/` | Environnement Python (à recréer, jamais versionné) |
