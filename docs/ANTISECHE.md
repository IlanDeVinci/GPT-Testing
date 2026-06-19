# Antisèche — projet, choix expliqués, glossaire (pour l'oral)

## 1. Le projet en 3 lignes
Jeu créatif en français « Histoires Trouées » : un **mini-GPT** invente une courte
histoire, on y perce des trous, un **mini-BERT** propose des mots pour les combler.
Les deux Transformers sont **entraînés localement, from scratch** (aucun poids externe).

## 2. Glossaire (tous les acronymes)
| Terme | Signification | En clair |
|---|---|---|
| **Transformer** | — | Architecture de réseau de neurones basée sur l'« attention » ; base de GPT et BERT. |
| **GPT** | *Generative Pre-trained Transformer* | Modèle **causal** : lit de gauche à droite, **prédit le mot suivant** → sert à générer du texte. |
| **BERT** | *Bidirectional Encoder Representations from Transformers* | Modèle **bidirectionnel** : voit le contexte des deux côtés → sert à remplir un trou. |
| **LM** | *Language Model* (modèle de langage) | Modèle qui apprend la probabilité du mot suivant (ou masqué). |
| **MLM** | *Masked Language Modeling* | Entraînement de BERT : on masque ~15 % des mots, il doit les deviner. |
| **from scratch** | « à partir de zéro » | Poids initialisés au hasard, entraînés sur NOTRE corpus (vs partir d'un modèle existant). |
| **token / tokenizer** | — | Un *token* = un morceau de mot. Le *tokenizer* découpe le texte en tokens. |
| **BPE** | *Byte Pair Encoding* | Méthode de tokenisation : fusionne les paires de caractères fréquentes. Vocabulaire 16 000. |
| **ByteLevel** | — | Variante de BPE qui encode chaque caractère en **octets UTF-8** (jamais de mot inconnu). |
| **perplexité (PPL)** | — | Mesure de qualité d'un LM : « à quel point le modèle est surpris » par le texte. **Plus bas = mieux.** |
| **NLL** | *Negative Log-Likelihood* | Log-vraisemblance négative ; la « loss » d'un LM. `perplexité = exp(NLL)`. |
| **accuracy** | exactitude | % de fois où le modèle devine le bon mot suivant. |
| **époque (epoch)** | — | Un passage complet sur tout le corpus. |
| **LR** | *Learning Rate* (taux d'apprentissage) | Taille des pas de l'optimiseur. Trop grand = instable/oubli ; trop petit = lent. |
| **scheduler cosine** | — | Fait **décroître** le LR en courbe cosinus jusqu'à ~0 → atterrissage en douceur. |
| **SFT** | *Supervised Fine-Tuning* | Fine-tuning supervisé : continuer l'entraînement sur des exemples ciblés. |
| **DPO** | *Direct Preference Optimization* | Aligner le modèle sur des **préférences** (paires « bonne/moins bonne réponse »). |
| **RLHF** | *Reinforcement Learning from Human Feedback* | Alignement par renforcement (plus lourd que DPO ; pas utilisé ici). |
| **RAG** | *Retrieval-Augmented Generation* | Récupérer des passages pertinents et s'en servir à la génération. |
| **AMP** | *Automatic Mixed Precision* | Calcul en 16 bits là où c'est possible → plus rapide, moins de mémoire. |
| **GPU / VRAM** | — | Carte graphique / sa mémoire (ici 8 Go), qui limite la taille des modèles. |
| **API** | *Application Programming Interface* | Le serveur (FastAPI) qui relie le modèle au jeu web. |
| **Chinchilla** | (loi d'échelle, DeepMind 2022) | Règle : un modèle est « bien nourri » avec ~**20 tokens par paramètre**. |

## 3. Les modèles
| Modèle | Rôle | Params |
|---|---|---|
| mini-GPT from-scratch (spécialisé narratif) | génère l'histoire | **33,7 M** |
| mini-BERT (MLM) | remplit les trous + encodeur RAG | 9,0 M |
| GPT-2 FR pré-entraîné (`asi/gpt-fr-cased-small`) | lignage de comparaison | 124,2 M |

**Pourquoi GPT pour générer et BERT pour remplir ?** Générer une suite, c'est
prédire le mot suivant de gauche à droite → c'est exactement un modèle **causal**
(GPT). Remplir un trou au milieu d'une phrase demande de regarder **avant ET après**
le trou → c'est un modèle **bidirectionnel** (BERT). Chaque tâche utilise le bon outil.

## 4. Les étapes (dans l'ordre)
1. **Données** : corpus FR nettoyé (Wikipédia + FineWeb + Wikisource) → 621 M tokens.
2. **Tokenizers** entraînés maison (BPE 16k pour le GPT, WordPiece pour le BERT).
3. **Pré-entraînement** du mini-GPT from scratch (prédiction du mot suivant).
4. **Spécialisation** sur un corpus narratif (continuation à faible LR).
5. **mini-BERT** entraîné en MLM.
6. **Application** : API FastAPI + interface React (+ option RAG).
7. **Évaluation** : perplexité par mot + accuracy, sur validation tenue à l'écart.

## 5. Les choix clés — quoi **et pourquoi en détail**

**① Normaliser la typographie, mais GARDER les accents.**
*Quoi* : remplacer `' « » … —` par leur équivalent ASCII 1 octet, mais conserver `é è à ç…`.
*Pourquoi* : le tokenizer ByteLevel encode chaque caractère en **octets UTF-8**. Les
caractères rares et multi-octets (apostrophe courbe = 3 octets, vue 14 677 fois) exigent
que le modèle ré-émette une séquence d'octets exacte ; un modèle sous-entraîné se trompe
→ octets invalides → caractère `�`. On supprime donc ce *piège* — mais pas les accents,
car (a) ils portent le sens (`a`/`à`, `ou`/`où`) et (b) ils sont assez fréquents pour être
appris correctement. On retire le bruit, pas l'information.

**② Qualité des données > volume.**
*Quoi* : on a écarté le corpus Gutenberg (pourtant 15× plus gros).
*Pourquoi* : un modèle apprend la **statistique** de ses données. 99 % de Gutenberg est du
français ancien **sans accents** (`etait` au lieu de `était`). L'ajouter aurait appris une
mauvaise orthographe. Plus de données *de mauvaise distribution* = modèle **pire**, pas
meilleur. Mieux vaut un petit corpus propre qu'un gros corpus dégradé.

**③ Dimensionner le modèle selon les données (Chinchilla).**
*Quoi* : on est passé d'un modèle de 97,7 M à un modèle de **34 M**.
*Pourquoi* : la loi de Chinchilla dit qu'il faut ~20 tokens par paramètre. Avec 621 M tokens,
l'optimum est ~31 M paramètres. Un modèle de 97,7 M est **affamé de données** : il a une
capacité qu'il ne peut pas remplir, donc il sous-apprend et gaspille le calcul. Comme on ne
peut pas facilement avoir plus de données sur un laptop, on **réduit le modèle** pour qu'il
soit bien nourri → il converge mieux sur le même corpus.

**④ Spécialisation de domaine plutôt que SFT/DPO classiques.**
*Quoi* : au lieu de fine-tuner agressivement, on **continue le pré-entraînement** sur un
corpus narratif avec un **LR faible** et une **validation narrative**.
*Pourquoi* : on a **mesuré** que le SFT puis le DPO *dégradaient* le modèle (perplexité qui
monte). Cause : un SFT à fort LR sur un petit corpus provoque l'**oubli catastrophique** (le
modèle écrase son français général pour coller au petit jeu). Aligner une base déjà faible,
c'est « polir une fondation fissurée ». Le LR faible nudge le modèle vers le domaine **sans
effacer** sa compétence de base ; on ne garde le checkpoint que s'il **améliore** la validation.

**⑤ Deux lignages (from-scratch ET pré-entraîné) pour comparer.**
*Quoi* : on a aussi fine-tuné un GPT-2 français existant.
*Pourquoi* : pour **mesurer** ce que le pré-entraînement apporte vraiment, en comparaison
contrôlée. C'est ce qui permet de prouver, chiffres à l'appui, que notre petit modèle
spécialisé bat le gros généraliste **sur le domaine du jeu**.

**⑥ RAG bâti sur notre propre BERT, en filet de sécurité.**
*Quoi* : recherche sémantique via les embeddings du mini-BERT ; sert surtout de secours.
*Pourquoi* : (a) réutiliser NOTRE BERT évite tout poids externe (cohérent avec « from
scratch ») et montre qu'un modèle sert à deux tâches. (b) On a constaté que **mettre des
passages en contexte distrait un bon modèle** (il s'éloigne de son propre style) → on garde
le RAG comme *filet* (proposer une vraie phrase humaine si la génération rate), pas comme
muselière. (c) On **centre** les embeddings (soustraire la moyenne) car ceux d'un BERT MLM
sont « anisotropes » (tout se ressemble, cosinus ~0,9) ; le centrage les rend discriminants.

## 6. Difficultés rencontrées
- Caractères `�` → diagnostic (tokenizer + sous-entraînement) et correction à la racine (choix ①).
- Modèle sous l'optimum (laptop) → assumé, puis corrigé en redimensionnant (choix ③).
- SFT/DPO contre-productifs → **prouvé par l'évaluation**, remplacés par la spécialisation (choix ④).
- Bug d'inférence (chargement concurrent des checkpoints → erreur « meta tensor ») → corrigé par des verrous.

## 7. LE résultat à retenir
Sur le **domaine narratif** (perplexité par mot, plus bas = mieux) :
- **Mon from-scratch spécialisé (33,7 M) : 230** 🥇 — accuracy 0,348
- GPT-2 FR pré-entraîné (124,2 M) : 267

→ **Un petit modèle spécialisé bat un gros généraliste sur son domaine, avec 3,7× moins de
paramètres.** Démonstration concrète des lois d'échelle + de la spécialisation.

## 8. Détails techniques : comment, précisément

**Récupération des données** (`scripts/download_french_corpus.py`)
On utilise la librairie **`datasets`** de Hugging Face en mode **streaming** (on lit
les exemples à la volée, sans télécharger le dataset entier). Trois sources, chacune
dans son fichier : `wikimedia/wikipedia` (`20231101.fr`), `HuggingFaceFW/fineweb-2`
(`fra_Latn`, web filtré/dédupliqué), `wikimedia/wikisource` (`20231201.fr`, littéraire).
Pour chaque document on **découpe en phrases** (regex sur `.!?`), on garde celles de
40 à 400 caractères, on déduplique. Puis `scripts/clean_data.py --inputs ...` fusionne
les sources, normalise (typo → ASCII, accents gardés, markup retiré) et met une partie
de côté pour la **validation** (1 ligne sur 50). → `data/clean-v3.txt`.

**Entraîner le BPE** (`scripts/train_tokenizer.py`, librairie `tokenizers`)
*BPE = Byte Pair Encoding* : on part des octets, et on **fusionne itérativement la
paire de symboles la plus fréquente** du corpus jusqu'à atteindre la taille de
vocabulaire (16 000). Réglages : normaliseur NFC, pré-tokenizer **ByteLevel** (encode
en octets UTF-8 → jamais de mot inconnu), `min_frequency=2`, tokens spéciaux
`[PAD] [UNK] [BOS] [EOS]`, et un post-traitement qui encadre par `[BOS] … [EOS]`.

**Qu'est-ce que WordPiece ?** (`scripts/train_bert_tokenizer.py`)
WordPiece est une tokenisation en sous-mots, comme BPE, **mais le critère de fusion
diffère** : au lieu de fusionner la paire la plus *fréquente*, elle fusionne celle qui
**augmente le plus la vraisemblance** du corpus (rapport fréquence-paire / fréquences
individuelles). Les morceaux de *continuation* d'un mot sont préfixés par `##`
(ex. « jardin » → `jard`, `##in`). C'est la tokenisation historique de BERT. Notre
réglage : normaliseur NFC + **minuscule** (BERT « uncased »), pré-tokenizer BERT,
vocab 16 000, tokens spéciaux `[PAD] [UNK] [CLS] [SEP] [MASK]`, encadrement `[CLS] … [SEP]`.

**Le fine-tuning / SFT** (`scripts/train_sft.py`)
On fabrique des paires **(début → suite)** depuis le corpus narratif : pour chaque
ligne, le *prompt* = la ligne, la *suite* = les 2-3 lignes suivantes. On entraîne le
modèle à prédire la suite, **avec la loss calculée UNIQUEMENT sur la suite** (les
tokens du prompt sont mis à `-100`, donc ignorés). Pour le lignage pré-entraîné on part
d'`asi/gpt-fr-cased-small` ; pour le from-scratch on reprend un checkpoint de base.

**Le DPO** — en deux temps
1. *Construire les préférences* (`scripts/build_dpo_data.py`) : pour chaque prompt, on
   génère **K=4 suites**, on les **note** avec une récompense = score du **juge BERT**
   (plausibilité de « prompt + suite » sous le MLM) + garde-fous (pénalités pour `[UNK]`,
   répétitions de 3-grammes, absence de ponctuation finale) + bonus si la suite réutilise
   des mots du prompt. On garde la **meilleure** (`chosen`) vs la **pire** (`rejected`).
2. *Entraîner* (`scripts/train_dpo.py`) : loss DPO codée à la main, avec un modèle de
   **référence figé** (copie du SFT). Formule :
   `L = -log σ( β·[ (logπ(chosen) − logπ_ref(chosen)) − (logπ(rejected) − logπ_ref(rejected)) ] )`,
   β = 0,1. Elle **augmente la marge** entre la bonne et la mauvaise suite sans trop
   s'éloigner de la référence. C'est un alignement par préférences **sans boucle de
   renforcement** (ni reward model séparé, ni PPO) — plus simple que le RLHF.

## 9. Questions probables du prof → réponses courtes
- **« C'est vraiment from scratch ? »** Oui : `GPT2Config` initialisé au hasard, tokenizers entraînés sur notre corpus. `from_pretrained` ne charge que NOS checkpoints locaux.
- **« Pourquoi la loss plafonne à ~3,4 ? »** C'est le plancher d'entropie pour un petit modèle sur du texte ouvert ; même un gros modèle ne descend guère plus bas. La métrique utile est la perplexité **par mot sur le domaine**, pas la loss absolue.
- **« Pourquoi ne pas battre le pré-entraîné sur le français général ? »** Il a vu ~100× plus de données. Limite d'échelle, pas défaut de code. On gagne là où on concentre la capacité : le narratif.
- **« Pourquoi pas juste `transformers` / le Trainer ? »** L'architecture EST `transformers` ; la boucle maison donne le même résultat. La coder fait partie de l'apprentissage.
- **« À quoi sert le mini-BERT ? »** Remplir les trous (`fill-mask`) ET encoder pour le RAG.
- **« Perplexité par mot vs par token ? »** Par token dépend du tokenizer (incomparable entre modèles). Par mot (`exp(NLL/nb_mots)`) est **comparable** entre lignages → c'est celle qu'on compare.
