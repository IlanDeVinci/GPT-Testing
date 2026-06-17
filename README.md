# Histoires Trouées

Petit jeu créatif en Python / PyTorch / Hugging Face côté modèles, FastAPI + React
côté application. Deux Transformers entraînés localement collaborent :

- un **mini-GPT** causal invente une courte histoire à partir d'un début de phrase ;
- un **mini-BERT** (masked language model) propose des mots pour combler les trous ;
- le joueur choisit une suggestion ou écrit son propre mot, puis découvre son
  histoire complète et un score d'originalité.

Aucun poids pré-entraîné externe n'est utilisé : tokenizers, GPT et BERT sont
entraînés sur un corpus local.

## Démarrage rapide

Backend (depuis la racine du projet) :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Frontend (dans un second terminal) :

```powershell
cd frontend
npm install
npm run dev
```

Le jeu s'ouvre sur `http://localhost:5173`. Le serveur Vite relaie `/api/*` vers le
backend sur le port `8000` (les modèles sont chargés à la première requête).

## API

| Méthode | Route        | Entrée                          | Sortie                                   |
|---------|--------------|---------------------------------|------------------------------------------|
| `GET`   | `/openings`  | —                               | débuts de phrase proposés                |
| `GET`   | `/models`    | —                               | modèles GPT disponibles + actif          |
| `POST`  | `/model`     | `{lineage}`                     | bascule le GPT actif                     |
| `POST`  | `/generate`  | `{opening?, n_blanks, seed?}`   | `{template, blanks[], story, opening}`   |
| `POST`  | `/fill`      | `{masked_text, top_k}`          | `{candidates:[{word, score}]}`           |

`template` contient des marqueurs `{{0}}`, `{{1}}`… pour les trous ; chaque entrée
de `blanks` fournit `answer` (mot d'origine), `hint` (catégorie devinée) et
`masked_text` (la phrase avec un seul `[MASK]`) prête pour `/fill`.

## Modèles entraînés

- mini-GPT actif : `checkpoints-story-v2` (tokenizer BPE `tokenizer-trained`) ;
- mini-BERT MLM : `tinybert/mlm-gutenberg-v3` (tokenizer WordPiece
  `tinybert/tokenizer-gutenberg-16k`).

Le backend résout automatiquement le dernier checkpoint via `latest.json`.

Le jeu démarre toujours sur le lignage `from-scratch`, c'est-à-dire le modèle
entraîné localement. Si un dossier `checkpoints-pre-sft` ou `checkpoints-pre-dpo`
existe, l'interface active aussi l'option `pretrained` dans le sélecteur. Le
checkpoint actif est affiché dans l'en-tête du jeu.

## Architecture

```text
project/
|-- backend/                  # API FastAPI, autonome
|   |-- main.py               # routes /generate, /fill, /openings
|   |-- generation.py         # chargement + inférence du mini-GPT
|   |-- mask_fill.py          # fill-mask du mini-BERT MLM
|   |-- blanks.py             # génération d'histoire + perçage des trous
|   `-- paths.py              # chemins des modèles entraînés
|-- frontend/                 # React + Vite + Framer Motion
|   `-- src/
|       |-- App.tsx           # machine à états du jeu
|       |-- api.ts
|       `-- components/
|-- scripts/                  # entraînement des modèles (voir ci-dessous)
|-- checkpoints-story-v2/     # mini-GPT
|-- tokenizer-trained/
`-- tinybert/
    |-- mlm-gutenberg-v3/
    `-- tokenizer-gutenberg-16k/
```

## Réentraîner / améliorer les modèles

Les scripts d'entraînement sont indépendants de l'application :

```powershell
python scripts/prepare_gutenberg_corpus.py

# mini-GPT
python scripts/train_tokenizer.py
python scripts/train_model.py

# mini-BERT MLM
python scripts/train_bert_tokenizer.py --corpus data/gutenberg/train.txt --output_dir tinybert/tokenizer-gutenberg-16k --vocab_size 16000
python scripts/train_tinybert_mlm.py --corpus data/gutenberg/train.txt --tokenizer_dir tinybert/tokenizer-gutenberg-16k --output_dir tinybert/mlm-gutenberg-v3
```

### Fine-tuning pré-entraîné en 45 minutes

Pour comparer ton modèle maison avec une base Hugging Face française sans lancer
une époque complète de plusieurs heures :

```powershell
python scripts/train_sft.py --hf_model asi/gpt-fr-cased-small --corpus data/clean-narratif.txt --output_dir checkpoints-pre-sft --epochs 1 --batch_size 4 --gradient_accumulation_steps 8 --learning_rate 5e-5 --num_workers 4 --max_length 256 --cont_sentences 2 --max_examples 60000 --max_steps 1500 --time_budget_minutes 44 --log_every 20
```

Le script sauvegarde un checkpoint proprement dès que `--max_steps` ou
`--time_budget_minutes` est atteint. `--max_examples` prend maintenant un
échantillon aléatoire du corpus, donc on ne réentraîne pas seulement sur le début
du fichier.

Voir les courbes pendant l'entraînement :

```powershell
python scripts/plot_training.py --metrics checkpoints-pre-sft/metrics.jsonl --live
```

Sauvegarder un PNG après l'entraînement :

```powershell
python scripts/plot_training.py --metrics checkpoints-pre-sft/metrics.jsonl --out outputs/pre-sft-curves.png
```

L'accuracy principale affichée par le SFT est maintenant la `top5_accuracy` :
elle vérifie si le bon token de continuation est dans les 5 prédictions les plus
probables. La `top1_accuracy` stricte reste loggée, mais elle est moins parlante
pour une tâche narrative où plusieurs suites peuvent être acceptables.

## Tests

```powershell
python -m pytest
```
