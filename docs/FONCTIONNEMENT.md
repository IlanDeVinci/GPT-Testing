# Fonctionnement du projet, de A à Z

Document de référence détaillé. Pour l'oral, voir [NOTE_PROF.md](NOTE_PROF.md).

---

## 1. Vue d'ensemble

« Histoires Trouées » est un jeu créatif en français qui met en scène **deux
Transformers entraînés localement** aux rôles complémentaires :

1. le **mini-GPT** (modèle causal) invente une courte histoire ;
2. on **perce des trous** dans cette histoire (on retire des mots) ;
3. le **mini-BERT** (masked language model) propose, pour chaque trou, une liste de
   mots probables ;
4. le joueur choisit une suggestion ou écrit son propre mot, puis découvre l'histoire
   complète et un **score d'originalité**.

L'idée pédagogique : un modèle imparfait devient un atout (suggestions surprenantes,
histoires absurdes) plutôt qu'un défaut.

---

## 2. Les deux modèles

### 2.1 Mini-GPT — modèle de langage causal
- **Architecture** : GPT-2 (`transformers.GPT2LMHeadModel`), version actuelle ~98 M
  paramètres (`n_embd=768`, `n_layer=12`, `n_head=12`, contexte `n_positions=512`).
- **Principe** : il lit une séquence de tokens et prédit **le token suivant**. En
  répétant cette prédiction, il génère du texte.
- **Entraînement** : objectif « causal LM » — minimiser la perte d'entropie croisée
  entre la prédiction et le vrai token suivant, sur chaque position.

### 2.2 Mini-BERT — masked language model
- **Architecture** : BERT (`transformers.BertForMaskedLM`), petit (~9 M paramètres,
  4–6 couches).
- **Principe** : on masque un mot (`[MASK]`) dans une phrase, il prédit le mot
  manquant **en regardant le contexte des deux côtés** (bidirectionnel).
- **Usage dans le jeu** : pour chaque trou, on lui envoie la phrase avec `[MASK]` et on
  récupère son **top-k** de mots les plus probables, avec leur score de confiance.

### 2.3 « From scratch » — preuve
On n'utilise **aucun poids pré-entraîné externe** :
- `GPT2Config(...)` puis `GPT2LMHeadModel(config)` créent des poids **aléatoires** ;
- de même `BertConfig(...)` → `BertForMaskedLM(config)` ;
- les **tokenizers** sont entraînés sur notre corpus (pas téléchargés) ;
- les appels `from_pretrained(...)` à l'inférence chargent **nos** dossiers de
  checkpoints locaux, pas un modèle de Hugging Face.

---

## 3. Pipeline de données

### 3.1 Sources (téléchargées en streaming via `datasets`)
| Source | Nature | Volume brut |
|---|---|---|
| OPUS Books | littérature traduite | 4 Mo |
| Wikipédia FR | encyclopédique | ~1 Go |
| FineWeb-2 (`fra_Latn`) | web filtré/dédupliqué | ~2 Go |
| Wikisource FR | textes du domaine public | ~63 Mo |

Total ≈ **3 Go de texte brut**. Script : `scripts/download_french_corpus.py`
(streaming → on ne télécharge pas les dumps complets de plusieurs dizaines de Go).

### 3.2 Nettoyage et normalisation — `scripts/clean_data.py`
Étapes, dans l'ordre, ligne par ligne :
1. **Normalisation Unicode NFC** + remplacement de la **typographie par de l'ASCII**
   (`'` → `'`, `«»` → `"`, `…` → `...`, `—` → `-`). On **garde les accents**.
2. **Filtres qualité** : longueur minimale, ponctuation finale obligatoire, rejet des
   lignes à trop forte proportion de symboles, et rejet des lignes **trop peu
   accentuées** (élimine le faux français sans accents).
3. **Déduplication** (optionnelle, `--no-dedup` pour les très gros corpus déjà
   dédupliqués comme FineWeb, pour ne pas saturer la RAM).
4. **Split validation** (`--val_output`) : 1 ligne conservée sur 50 (~2 %) est mise de
   côté pour mesurer le modèle sur des données qu'il n'a pas vues.
5. **Écriture en streaming** : on écrit au fur et à mesure (pas d'accumulation du
   corpus entier en mémoire).

Résultat : ~620 M tokens propres (≈ 18,5 M lignes d'entraînement + 0,38 M validation).

### 3.3 Pourquoi la normalisation typographique (le bug des `�`)
Le tokenizer du GPT est un **BPE ByteLevel** : il représente chaque caractère par ses
**octets UTF-8**. Un caractère multi-octets rare (apostrophe courbe `'` = 3 octets) est
**fragile** : un modèle sous-entraîné produit une séquence d'octets invalide à la
génération → le décodeur affiche `�`. En normalisant ces caractères vers leur version
ASCII 1 octet, on supprime la source du problème sans perdre d'information (les accents,
eux, sont fréquents et donc bien appris — on les garde).

### 3.4 Tokenisation
- **GPT** : BPE ByteLevel, vocabulaire 16 000 (`scripts/train_tokenizer.py`,
  bibliothèque `tokenizers`).
- **BERT** : WordPiece (`scripts/train_bert_tokenizer.py`).
- **Cache de tokens** : à l'entraînement, le corpus est tokenisé **une seule fois** et
  stocké en `uint16` sur disque, lu ensuite par **memory-mapping** (voir §4.2).

---

## 4. Entraînement — `scripts/train_model.py`

### 4.1 Boucle (codée à la main)
- optimiseur **AdamW**, **scheduler cosinus avec warmup** (`transformers`),
- **precision mixte (AMP)** : `autocast` + `GradScaler` → calcul en fp16, plus rapide
  et moins gourmand en mémoire,
- **accumulation de gradient** : simule un grand batch en additionnant les gradients
  de plusieurs petits batchs (utile à VRAM limitée),
- **clipping de gradient** (norme max 1.0) pour la stabilité,
- **checkpoints** sauvegardés à chaque époque + `latest.json`.

### 4.2 Dataset en lecture paresseuse (`LazyBlockDataset`)
Plutôt que charger tout le corpus tokenisé en RAM, on :
1. tokenise le corpus en streaming par morceaux → fichier `.bin` (`uint16`) sur disque ;
2. le lit en **`numpy.memmap`** : les blocs de 384 tokens sont découpés à la volée,
   sans tout charger en mémoire.
→ RAM quasi nulle même sur des centaines de millions de tokens, et compatible avec
plusieurs workers du `DataLoader`.

### 4.3 Métriques (codées à la main)
- **loss** (entropie croisée) ;
- **accuracy de prédiction du token suivant** : proportion de positions où
  `argmax(logits) == vrai token` ;
- écrites en continu dans `metrics.jsonl` → visualisées par `scripts/plot_training.py`
  (courbes live pendant l'entraînement, ou figées après).

### 4.4 Confort
- **Sauvegarde sur Ctrl-C** : interrompre l'entraînement sauvegarde un checkpoint
  `checkpoint-interrupted-step-N` au lieu de tout perdre.
- **`--max_steps`** : limite le nombre de mises à jour, et le scheduler décroît
  proprement sur ce budget (utile quand on n'a pas le temps d'une époque complète).

### 4.5 Évaluation — `scripts/evaluate_perplexity.py`
Sur le set de validation tenu à l'écart : **perplexité** (`exp(loss)`, plus bas =
mieux) et **accuracy**. Mesures comparables entre modèles de tailles différentes.

---

## 5. L'application (le jeu)

### 5.1 Backend — FastAPI (`backend/`)
- `generation.py` : charge le mini-GPT et génère le texte.
- `mask_fill.py` : charge le mini-BERT MLM et renvoie le top-k pour un `[MASK]`.
- `blanks.py` : génère une histoire puis y perce des trous (choix de mots-contenu
  espacés + indice grammatical heuristique).
- `main.py` : routes `/generate`, `/fill`, `/openings`, `/health`.
- `paths.py` : résout automatiquement les derniers modèles (bascule sur le nouveau
  modèle dès qu'un entraînement produit `checkpoints-v3/latest.json`).

### 5.2 Frontend — React + Vite (`frontend/`)
React + TypeScript + **Framer Motion** (animations) + CSS maison. Affiche les deux
modèles à l'œuvre : barres de confiance du BERT, slot actif, révélation animée de
l'histoire, score d'originalité.

---

## 6. Récapitulatif : librairies vs code maison

### Bibliothèques utilisées
| Librairie | Rôle |
|---|---|
| PyTorch (`torch`) | tenseurs, autograd, AdamW, AMP, DataLoader |
| HF `transformers` | architectures GPT-2 / BERT, tokenizer rapide, scheduler cosinus |
| HF `tokenizers` | entraînement des tokenizers BPE et WordPiece |
| HF `datasets` | téléchargement streaming des corpus |
| NumPy | cache de tokens `uint16` en memory-map |
| FastAPI + uvicorn | API du jeu |
| React + Vite + Framer Motion | interface |
| matplotlib | courbes d'entraînement |

### Codé par nous
| Composant | Fichier |
|---|---|
| Nettoyage + normalisation + split validation | `scripts/clean_data.py` |
| Téléchargeur multi-sources | `scripts/download_french_corpus.py` |
| Dataset memmap + cache streaming | `scripts/train_model.py` |
| Boucle d'entraînement (AMP, accumulation, accuracy, log, Ctrl-C) | `scripts/train_model.py` |
| Évaluation perplexité / accuracy | `scripts/evaluate_perplexity.py` |
| Visualisation des courbes | `scripts/plot_training.py` |
| Logique du jeu (génération + trous + fill-mask + API) | `backend/` |
| Interface complète | `frontend/` |

---

## 7. Décisions et leçons techniques

1. **`�` = octets UTF-8 invalides** d'un ByteLevel sous-entraîné → normalisation ASCII
   en gardant les accents (cause traitée à la racine).
2. **Qualité > volume** : Gutenberg écarté car ~99 % sans accents (mesuré par un filtre
   quantitatif).
3. **Lois d'échelle (Chinchilla, ~20 tokens/paramètre)** : un 98 M veut ~2 Md tokens ;
   on en a ~620 M → entraînement partiel assumé, LR décroissant sur le budget.
4. **8 Go VRAM** : AMP + gradient checkpointing + dataset paresseux. Leçon debug : sous
   Windows (WDDM), dépasser la VRAM ne provoque pas une erreur mais un **débordement
   sur la RAM système** (~100× plus lent) — d'où des ralentissements quand plusieurs
   process se partagent le GPU.
5. **Boucle maison plutôt que `transformers.Trainer`** : pour le contrôle total
   (métriques, Ctrl-C, log custom) et la compréhension. `Trainer` serait plus court
   mais une boîte noire.

---

## 8. Suites prévues
1. **SFT** (fine-tuning supervisé) sur un style « histoires courtes ».
2. **DPO** (optimisation par préférences) en utilisant le **juge de cohérence** comme
   labelleur automatique — alternative stable au RLHF complet à cette échelle.
3. Réentraîner le mini-BERT sur le nouveau corpus propre pour de meilleures suggestions.
