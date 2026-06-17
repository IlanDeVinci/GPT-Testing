# Historique du projet — de l'idée au jeu « Histoires Trouées »

Résumé chronologique de ce qui a été fait, pour s'y retrouver.

## 1. Point de départ

Deux Transformers entraînés localement, à valoriser dans un jeu créatif :
- un **mini-GPT** causal (~13,8 M params) qui génère du texte ;
- un **mini-BERT** entraîné en MLM (masked language model).

Un premier jeu existait (« Le Gardien du récit », interface Streamlit) où le GPT
proposait des suites et un classifieur de cohérence les notait. Il ne convenait pas.

## 2. Nouveau concept retenu — Mad Libs IA

Jeu **« Histoires Trouées »** : le mini-GPT invente une courte histoire, on y perce
des trous, et le mini-BERT (en `fill-mask`) propose des mots pour chaque trou. Le
joueur choisit une suggestion ou écrit le sien, puis découvre son histoire et un
score d'originalité.

Idée clé : un modèle « pas parfait » devient un atout (suggestions surprenantes,
histoires absurdes) au lieu d'un défaut.

## 3. Backend (FastAPI) — `backend/`

- `mask_fill.py` : charge le checkpoint MLM (`tinybert/mlm-gutenberg-v3`) et renvoie
  le top-k de mots pour un `[MASK]`.
- `blanks.py` : génère une histoire avec le GPT, puis perce des trous (mots-contenu
  espacés, indice grammatical heuristique).
- `generation.py`, `paths.py` : chargement du mini-GPT, chemins des modèles.
- `main.py` : routes `/generate`, `/fill`, `/openings`, `/health`.

## 4. Frontend (React + Vite + Framer Motion) — `frontend/`

Interface soignée : badges des deux modèles, slot actif en surbrillance, barres de
confiance du mini-BERT, révélation animée de l'histoire, score d'originalité.

## 5. Découplage et nettoyage

- Le backend a été rendu **autonome** : il ne dépend plus de l'ancien jeu (vérifié en
  bloquant l'import). `app.py` et `story_game/` ont été supprimés.
- Les modèles et données **inutilisés** (anciens checkpoints, classifieurs du juge,
  MLM obsolètes, données de cohérence, `outputs/`) ont été déplacés dans `archives/`
  (rien n'a été supprimé).
- Modèles actifs conservés : `checkpoints-story-v2`, `tokenizer-trained`,
  `tinybert/mlm-gutenberg-v3`, `tinybert/tokenizer-gutenberg-16k`.

## 6. Lancement

- `start.ps1` : lance backend + frontend dans deux fenêtres et ouvre le jeu.
- `stop.ps1` : arrête les deux serveurs.

## 7. Amélioration du mini-GPT (en cours)

Détail et justifications dans [AMELIORATION_GPT.md](AMELIORATION_GPT.md).

- **Diagnostic** : les `�` viennent du tokenizer ByteLevel + d'un modèle
  sous-entraîné qui rate les caractères multi-octets (14 677 apostrophes courbes
  dans OPUS).
- **Données** : `clean_data.py` normalise la typographie vers l'ASCII en gardant les
  accents. Le corpus Gutenberg a été écarté (filtre d'accents : ~99 % sans accents).
- **Volume** : `download_french_corpus.py` télécharge du français propre (Wikipédia
  FR) pour dépasser la petite taille d'OPUS, puis on fusionne et normalise.
- **Réentraînement v3** : nouveau tokenizer + entraînement plus long, avant toute
  technique d'alignement (SFT/DPO/RLHF), qui viendront ensuite.

## Documents liés

- [README.md](../README.md) — installation et utilisation.
- [AMELIORATION_GPT.md](AMELIORATION_GPT.md) — diagnostic et choix d'entraînement.
