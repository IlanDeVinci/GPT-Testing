# Dossier de rendu — « Histoires Trouées »

Jeu créatif en français propulsé par **deux Transformers entraînés localement** :
un mini-GPT qui invente une histoire, un mini-BERT qui propose des mots pour
combler des trous. Ce document récapitule la démarche, les résultats chiffrés,
ce qu'il faut dire à l'oral, et le plan d'amélioration.

---

## 1. Chronologie — de l'idée au jeu

### Phase A — Cadrage et recherche
1. **Point de départ** : deux Transformers entraînés localement à valoriser. Un
   premier jeu (« Le Gardien du récit », Streamlit, où un classifieur notait la
   cohérence des suites du GPT) ne convainquait pas.
2. **Question de recherche** : comment rendre *utile* un petit modèle imparfait,
   entraînable sur un laptop ?
   **Réponse retenue** : un **Mad Libs IA** — le GPT invente une histoire, on y
   perce des trous, le BERT (`fill-mask`) propose des mots. L'imperfection du
   modèle devient un ressort ludique au lieu d'un défaut.

### Phase B — Données (le cœur du travail)
3. Corpus initial : **OPUS Books** (en-fr), 25 000 lignes propres.
4. **Diagnostic des caractères `�`** : le tokenizer ByteLevel encode en octets
   UTF-8 ; un modèle sous-entraîné rate les caractères multi-octets (14 677
   apostrophes courbes dans le corpus). → **Réponse** : `clean_data.py` normalise
   la typographie vers l'ASCII (`'→'`, `«»→"`, `…→...`) **en gardant les accents**
   (ils portent le sens : `a`/`à`, `ou`/`où`).
5. **Décision « qualité > volume »** : Gutenberg testé puis **écarté** (un filtre
   sur le taux d'accents rejette ~99 % du corpus : français ancien sans accents).
   Corpus élargi avec du français propre : **Wikipédia FR (1,1 Go) + FineWeb-FR
   (2,0 Go) + Wikisource (63 Mo)** → `data/clean-v3.txt` = **2,4 Go ≈ 621,6 M tokens**.

### Phase C — Entraînement, lignage « from scratch »
6. Tokenizer **BPE** entraîné maison (`tokenizer-v3`, vocabulaire 16 000).
7. **Pré-entraînement** d'un GPT-2 *from scratch* (**97,7 M paramètres**, 12
   couches, d=768, contexte 512) → `checkpoints-v3`.
8. **SFT** (fine-tuning supervisé) sur corpus narratif → `checkpoints-sft`.
9. **DPO** (optimisation par préférences), paires construites avec le juge de
   cohérence (`build_dpo_data.py` + `bert_reward.py`) → `checkpoints-dpo`.

### Phase D — Entraînement, lignage « pré-entraîné » (comparatif)
10. Base française **`asi/gpt-fr-cased-small`** (124,2 M) → **SFT**
    `checkpoints-pre-sft` → **DPO** `checkpoints-pre-dpo`.

### Phase E — BERT et application
11. Mini-BERT MLM entraîné (**9,0 M paramètres**, 6 couches) → `tinybert/mlm-v3`
    (loss 2,62).
12. **Backend FastAPI** autonome (`/generate`, `/fill`, `/model`, `/openings`) +
    **frontend React/Vite**. Sélecteur de lignage GPT dans l'interface.
13. Outillage : évaluation (`eval_all.py`, `evaluate_perplexity.py`), graphes
    (`plot_training.py`), sauvegarde sur Ctrl-C, reprise complète (optimiseur +
    scheduler), checkpoints suivis par `latest.json`.

---

## 2. Bibliothèques vs code maison

- **Librairies** : PyTorch (autograd, optimiseur), Hugging Face `transformers`
  (architectures GPT-2 / BERT), `tokenizers`, `datasets`.
- **Codé par nous** : toute l'orchestration — nettoyage/normalisation des
  données, entraînement des tokenizers, boucle d'entraînement (precision mixte
  AMP, accumulation de gradient, scheduler cosine, checkpoints, accuracy,
  reprise sans couture), pipeline de tokenisation en cache *memory-mapped*,
  évaluation, et tout le jeu.
- **From scratch** = aucun poids externe : initialisation aléatoire
  (`GPT2Config → GPT2LMHeadModel`), tokenizers entraînés sur notre corpus. Les
  `from_pretrained` ne chargent que **nos** checkpoints locaux.

---

## 3. Résultats chiffrés (évaluation sur held-out)

Évaluation sur les premiers 1,2 M caractères de `data/clean-v3-val.txt`
(190 433 mots), même texte et même `block_size` pour tous les modèles.
Script : `scripts/eval_all.py` — graphe : `outputs/eval_all.png`.

> ⚠️ La **perplexité par token** n'est PAS comparable entre lignages (les deux
> tokenizers ne découpent pas pareil). La métrique honnête inter-lignages est la
> **perplexité par mot** (`exp(NLL_total / nb_mots)`).

| Modèle | Params | Perplexité / mot ↓ | Perplexité / token | Accuracy ↑ |
|---|---|---|---|---|
| From-scratch — pré-entraîné (`v3`)   | 97,7 M  | **364,8** | 38,0 | 0,331 |
| From-scratch — SFT (`sft`)           | 97,7 M  | 842,1 | 63,6 | 0,276 |
| From-scratch — DPO (`dpo`)           | 97,7 M  | 879,6 | 65,3 | 0,272 |
| Pré-entraîné FR — SFT (`pre-sft`)    | 124,2 M | **355,4** | 56,0 | 0,326 |
| Pré-entraîné FR — DPO (`pre-dpo`)    | 124,2 M | 373,7 | 58,0 | 0,326 |

**Trois conclusions importantes (à montrer au prof) :**

1. **Le SFT puis le DPO ont DÉGRADÉ le modèle from-scratch** : la perplexité par
   mot passe de 365 → 842 → 880, et l'accuracy chute de 0,33 → 0,27. Le SFT
   (sur un corpus narratif étroit, LR trop fort, sans early-stopping sur la
   validation) a provoqué un **oubli catastrophique** ; le DPO sur une base déjà
   abîmée n'a fait qu'empirer. C'est l'explication directe du « pas glorieux ».
2. **Le DPO dégrade dans les deux lignages** (FS 842→880, pré-entraîné 355→374).
   Sur ces petits modèles et avec si peu de paires, il n'apporte rien.
3. **Le lignage pré-entraîné est plus stable** : la base française résiste mieux
   au fine-tuning (accuracy maintenue à 0,326). C'est lui qui donne le rendu le
   plus lisible pour la démo.

> Le jeu chargeait jusqu'ici `checkpoints-dpo` (le **pire** des checkpoints
> from-scratch, ppl 880). C'est corrigé : `backend/paths.py` privilégie désormais
> `checkpoints-v4` (le nouveau modèle, voir §6) puis le meilleur disponible.

### Graphes d'entraînement (dossier `outputs/`)
- `curve-v3.png`, `curve-sft.png`, `curve-pre-sft.png`, `curve-dpo.png`,
  `curve-pre-dpo.png` — loss + accuracy par run (accuracy en une seule ligne lissée).
- `compare-fromscratch.png` — superposition pré-entraînement / SFT / DPO.
- `eval_all.png` — comparatif final perplexité par mot + accuracy.

---

## 4. Script oral pour le prof (~4 min)

> **Le projet.** « Histoires Trouées », un jeu créatif en français. Deux
> Transformers que j'ai entraînés moi-même collaborent : un mini-GPT (97,7 M
> paramètres, architecture GPT-2) invente une courte histoire ; un mini-BERT
> (9 M paramètres, masked language model) propose des mots pour combler des trous.
>
> **Librairies vs maison.** J'utilise PyTorch et Hugging Face pour les briques
> (architecture, optimiseur). Mais j'ai codé toute l'orchestration : nettoyage
> des données, tokenizers entraînés sur mon corpus, boucle d'entraînement
> (precision mixte, accumulation de gradient, scheduler, reprise), l'évaluation
> et tout le jeu. *From scratch* = j'initialise le GPT aléatoirement et je
> l'entraîne sur mon propre texte, aucun poids externe.
>
> **Démarche.** J'ai suivi l'ordre des leviers par retour sur investissement :
> données propres → pré-entraînement → SFT → DPO. J'ai construit **deux lignages
> comparables** (un 100 % maison, un partant d'un modèle français pré-entraîné)
> pour mesurer ce qu'apporte vraiment le pré-entraînement.
>
> **Les difficultés rencontrées** (la partie qui montre la compréhension) :
> - **Caractères corrompus `�`** : diagnostiqués comme un tokenizer ByteLevel sur
>   un modèle sous-entraîné. Réglés à la racine par normalisation typographique,
>   sans perdre les accents.
> - **Qualité vs volume** : j'ai voulu ajouter Gutenberg (15× plus gros), mais un
>   filtre quantitatif a montré 99 % de français ancien sans accents — écarté
>   pour ne pas apprendre une mauvaise orthographe.
> - **Lois d'échelle (Chinchilla)** : mon corpus fait 621 M tokens, ce qui suffit
>   pour un modèle d'~31 M paramètres ; mon modèle de 97,7 M est donc **3× trop
>   gros pour mes données** et reste sous-entraîné. C'est ma principale limite.
> - **SFT/DPO contre-productifs** : l'évaluation chiffrée montre que mon
>   fine-tuning a *dégradé* le from-scratch (perplexité 365 → 880). J'ai
>   diagnostiqué un oubli catastrophique : aligner une base trop faible empire
>   les choses. Leçon : on corrige d'abord la base.
> - **Contrainte matérielle (8 Go VRAM)** : precision mixte + dataset
>   memory-mapped en lecture paresseuse pour ne pas saturer la mémoire.
> - **Un bug d'inférence** : chargement intermittent des checkpoints (les poids
>   « liés » ne sont pas stockés en safetensors → erreur *meta tensor* sur GPU).
>   Diagnostiqué et corrigé en matérialisant les poids avant le passage sur GPU.
>
> **Honnêteté sur les résultats.** Les textes ne sont pas parfaitement fluides :
> c'est cohérent avec un modèle entraîné sous l'optimum sur du matériel grand
> public. Mon mesurage (perplexité par mot, comparable entre tokenizers) le
> prouve plutôt que de le supposer. Prochaine étape : un modèle redimensionné à
> ~34 M, bien nourri par mes 621 M tokens.

---

## 5. Le bug d'interface (corrigé)

**Symptôme :** `NotImplementedError: Cannot copy out of meta tensor` à l'ouverture
d'un modèle (erreurs 500 intermittentes sur `/fill` et `/model`).

**Cause :** les checkpoints safetensors ne stockent pas les poids *liés*
(`lm_head.weight` du GPT, `cls.predictions.decoder.weight` du BERT — ils
référencent l'embedding d'entrée). Avec transformers 4.55 + torch 2.7, le poids
manquant est recréé sur le *meta device* ; le `.to("cuda")` enchaîné échoue par
intermittence.

**Correctif** (`backend/mask_fill.py`, `backend/generation.py`) :
`from_pretrained(..., low_cpu_mem_usage=False)` (matérialise tout en RAM) +
`tie_weights()` (relie proprement) + `.to(device)`. Vérifié 10/10 chargements OK.
Ce bug n'affectait pas la qualité du texte (les poids étaient correctement liés).

---

## 6. Plan d'amélioration

Par retour sur investissement décroissant.

### ① Entraîner un modèle bien dimensionné (le levier n°1) — *à lancer ce soir*
Le diagnostic Chinchilla est sans appel : 621 M tokens → optimum à ~31 M
paramètres. Script prêt :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_tonight.ps1
```

Il entraîne un GPT **~33,7 M** (8 couches, d=512, contexte 512) pendant
14 000 pas (~344 M tokens vus, ~10 tokens/paramètre), scheduler cosine décroissant
jusqu'à ~0, sortie `checkpoints-v4`. Attention **SDPA + TF32** activées
automatiquement (3× plus rapide) → ~8–9 h, soit une nuit.
Un checkpoint est sauvegardé à la fin et sur Ctrl-C (jamais de travail perdu).
En fin de run il génère la courbe (`outputs/curve-v4.png`) et relance
l'évaluation comparative. Le jeu utilisera `checkpoints-v4` automatiquement
(priorité ajoutée dans `backend/paths.py`). Avec un week-end devant toi, monte
`--max_steps` à 25 000 pour une époque complète.

### ② Ne pas réappliquer SFT/DPO tel quel
L'évaluation montre qu'ils dégradent. Si tu veux spécialiser le style narratif :
SFT **doux** (LR ~2e-5, peu de pas) avec **early-stopping sur la perplexité de
validation**, et garde le checkpoint seulement s'il *améliore* la validation.
Abandonne le DPO tant que la base n'est pas solide (il n'a jamais aidé ici).

### ③ Pour une démo immédiatement plus lisible
En attendant le ré-entraînement, bascule le jeu sur le **lignage pré-entraîné**
(le plus stable) : dans `backend/main.py`, `_active_lineage = "pretrained"`.

### ④ Réglages d'inférence (gratuit)
Pour un petit modèle, baisser `temperature` à ~0,7 et `top_p` à ~0,85 dans
`backend/generation.py` réduit les dérapages (un peu moins de diversité).

### ⑤ Mesurer systématiquement
Avant/après chaque changement : `python scripts/eval_all.py` (perplexité par mot
+ accuracy) et lecture humaine de ~10 générations à température fixe. Sans ces
chiffres, « mieux » reste subjectif.

### ⑥ Rendre le from-scratch compétitif : la spécialisation de domaine
On ne peut pas battre un modèle pré-entraîné sur des **dizaines de milliards** de
tokens (le nôtre en a 621 M) sur le français *général* — c'est une limite
d'échelle, pas un défaut de code. **Mais** un petit modèle dont toute la capacité
est concentrée sur le domaine du jeu (les histoires) peut le battre **sur ce
domaine**. Démarche (≠ le SFT agressif qui avait tout cassé) :

1. Corpus narratif propre + validation tenue à l'écart (déjà construit) :
   ```powershell
   python scripts/clean_data.py --inputs data/clean-narratif.txt data/wikisource-fr.txt data/clean.txt `
       --output data/narratif-train.txt --val_output data/narratif-val.txt --val_every 50 --min_chars 40
   ```
   → ~15 M tokens narratifs uniques (train) + `data/narratif-val.txt` (held-out).
2. **Adaptation de domaine** : on reprend les *poids* de la base from-scratch
   (v4) et on continue l'entraînement en LM causal sur le narratif avec un **LR
   faible (3e-5) + optimiseur neuf** (`--fresh_optimizer`). Faible LR = adaptation
   sans oubli catastrophique. Le modèle reste 100 % from-scratch.
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\train_narratif.ps1
   ```
   Sortie `checkpoints-narratif` (≈ 1 h, checkpoint par époque).
3. **Le test qui compte** : `eval_all.py` sur `data/narratif-val.txt`. Si le
   from-scratch spécialisé a la perplexité par mot la plus basse **sur le
   narratif**, tu as gagné sur ton domaine — résultat fort et honnête pour le
   rendu. Le jeu l'utilise alors automatiquement (priorité dans `paths.py`).

À lancer **après** `train_tonight.ps1` (pour partir de v4) ; sinon il part de v3.

### ⑦ RAG : récupération bâtie sur notre propre mini-BERT
Un module de récupération sémantique, **sans aucun poids externe** : le mini-BERT
sert d'encodeur (mean-pooling de ses états cachés, centrage « all-but-the-mean »
pour réduire l'anisotropie), on indexe le corpus narratif et on cherche par
cosinus.

- **Conditionnement de style** : on récupère des passages proches du début et on
  les met en contexte du mini-GPT (`make_story(..., retriever=...)`).
- **Filet de sécurité** : si la génération est vide/trop courte, on propose une
  vraie phrase humaine du corpus → texte toujours lisible dans le jeu.

```powershell
python scripts/build_rag_index.py --corpus data/narratif-train.txt   # une fois
python scripts/demo_rag.py                                           # démo avant/après
```
API : `POST /retrieve {query, top_k}` ; `POST /generate {opening, use_rag:true}`.

**Honnêteté (à dire au prof)** : le RAG ne corrige PAS la fluidité (elle vient des
poids, pas du contexte) ; il améliore la pertinence/le style et garantit un
fallback lisible. Les embeddings d'un BERT entraîné en MLM sont moyens — le
centrage aide nettement (scores plus discriminants), mais ça reste un encodeur
modeste. C'est un usage juste et défendable de la récupération, pas une rustine
sur un modèle faible.

### Note sur « ne répond pas aux questions »
C'est attendu : ce ne sont pas des chatbots. Le GPT *continue* un texte, le BERT
*complète* un trou. Pour qu'il « réponde », il faudrait un format
instruction→réponse au SFT — un autre projet. Ici la bonne métrique est la
**cohérence/fluidité**, pas le question-réponse.
