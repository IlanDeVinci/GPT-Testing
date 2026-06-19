# Migration vers un autre PC (GPU plus rapide)

Deux canaux : **le code par Git** (léger), **les gros fichiers à la main** (corpus +
modèles), parce qu'ils dépassent la limite GitHub (100 Mo/fichier).

## Principe

Le nouveau PC va **ré-entraîner** (v4 puis narratif). Il n'a donc PAS besoin des
anciens checkpoints ni des corpus bruts. On ne transfère que :
- le **corpus nettoyé** nécessaire à l'entraînement,
- les **tokenizers**,
- le **mini-BERT déjà entraîné** (pour le RAG et le jeu),
- les **artefacts du rendu** (graphes/JSON, quelques Mo).

On NE transfère PAS (régénérable ou inutile) :
- `.venv-gpu/` (à recréer), `frontend/node_modules/` (npm install),
- les bruts `data/fineweb-fr.txt` (2,0 Go), `data/wikipedia-fr.txt` (1,1 Go), `data/raw.txt`,
- les caches `data/*.tokens.bin` (régénérés au 1er entraînement),
- `data/rag-index/` (reconstruit par `build_rag_index.py`),
- les vieux checkpoints from-scratch (`v3`, `sft`, `dpo`) — le nouveau PC fait mieux.

### Ce qu'on transfère (~2,8 Go)

| Fichier / dossier | Taille | Pourquoi |
|---|---|---|
| `data/clean-v3.txt` | 2 435 Mo | corpus de pré-entraînement (base v4) |
| `data/clean-bert.txt` | 215 Mo | corpus du mini-BERT (si re-entraînement) |
| `data/clean-v3-val.txt` | 50 Mo | éval perplexité (général) |
| `data/narratif-train.txt` | 54 Mo | spécialisation narrative |
| `data/narratif-val.txt` | 2 Mo | éval narrative (le test qui compte) |
| `tokenizer-v3/` | 2 Mo | tokenizer du GPT |
| `tinybert/tokenizer-v3/` | 1 Mo | tokenizer du BERT |
| `tinybert/mlm-v3/` | 104 Mo | mini-BERT entraîné (RAG + jeu) |
| `outputs/` | qq Mo | graphes + JSON du rendu |

Optionnel (`-IncludeCheckpoints`) si tu veux garder la comparaison du lignage
pré-entraîné, non re-dérivable exactement : `checkpoints-pre-sft` (958 Mo),
`checkpoints-pre-dpo` (958 Mo).

## Étape 1 — Code par Git

Le `.gitignore` a été corrigé pour exclure venv / data / checkpoints / modèles.

```powershell
git init
git add .
git status            # VERIFIE : aucun .txt enorme, aucun checkpoint, pas de .venv-gpu
git commit -m "Projet Histoires Trouees : code + scripts"
git branch -M main
git remote add origin https://github.com/<toi>/<repo>.git
git push -u origin main
```

Sur le nouveau PC : `git clone https://github.com/<toi>/<repo>.git`

## Étape 2 — Gros fichiers à la main

Sur CE PC, prépare un dossier `transfer/` prêt à copier :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_for_transfer.ps1
# avec les checkpoints pre-entraines en plus :
powershell -ExecutionPolicy Bypass -File scripts\package_for_transfer.ps1 -IncludeCheckpoints
```

Puis copie `transfer/` vers le nouveau PC par le moyen le plus rapide :
- **Disque externe / clé USB** (le plus rapide pour 2,8 Go) ;
- **Google Drive / OneDrive** (gratuit, > 2 Go ; WeTransfer gratuit plafonne à 2 Go) ;
- **Réseau local** : `robocopy \\AUTRE-PC\partage\project transfer /E`.

Sur le nouveau PC, copie le contenu de `transfer/` DANS le dossier du repo cloné
(il recrée `data/`, `tokenizer-v3/`, `tinybert/...`, `outputs/`).

## Étape 3 — Installer l'environnement sur le nouveau PC

> **PRÉREQUIS : Python 3.10 minimum.** Le code utilise la syntaxe `str | None` /
> `list[...]` (PEP 604, 3.10+). Avec un Python 3.9 ou antérieur tu obtiens
> `unsupported operand type for |: type and NoneType`. Vérifie avec
> `python --version` ; sinon installe 3.10+ (python.org) et crée le venv avec
> `py -3.12 -m venv .venv-gpu` (le launcher `py` choisit la version).

```powershell
py -3.12 -m venv .venv-gpu      # 3.12, 3.11 ou 3.10 — PAS 3.9
.\.venv-gpu\Scripts\Activate.ps1
python --version                # doit afficher 3.10+
# IMPORTANT : installe torch avec le CUDA de TON GPU (sinon il tourne sur CPU).
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

(Adapte `cu126` à ta version CUDA ; voir https://pytorch.org/get-started/locally/.)

## Étape 4 — Lancer

```powershell
# Sur le nouveau GPU (plus rapide), tu peux viser plus que les 14000 pas du laptop :
powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1            # tout, graphes en direct
# Le cache de tokens et l'index RAG se (re)construisent automatiquement.
```

Si le GPU est nettement plus rapide, augmente `--max_steps` dans
`scripts\train_tonight.ps1` (ou `train_all.ps1`) pour viser une époque complète
(~25000 pas) et un modèle mieux entraîné.

## Vérification anti-bêtise avant `git push`

- `git status` ne doit lister AUCUN fichier > 100 Mo.
- Si un gros fichier apparaît : il manque au `.gitignore`, corrige-le AVANT de commit.
- Jamais `git add -f` sur un fichier de données ou un checkpoint.
