# Note orale — à dire au prof

À lire en ~3 minutes. Détails complets dans [FONCTIONNEMENT.md](FONCTIONNEMENT.md).

## Le projet en une phrase
Un jeu créatif en français, « Histoires Trouées », propulsé par **deux Transformers
entraînés localement from scratch** : un mini-GPT qui invente une histoire, et un
mini-BERT qui propose des mots pour combler des trous.

## Ce qu'on a construit
- Un **mini-GPT** (architecture GPT-2, ~98 M paramètres) : modèle de langage causal,
  prédit le mot suivant → génère du texte.
- Un **mini-BERT** (masked language model) : prédit un mot masqué `[MASK]` → propose
  des suggestions pour chaque trou.
- Une **application web** : API FastAPI + interface React.

## Bibliothèques vs code maison (point important)
- **Des librairies** : l'architecture des modèles et les briques de bas niveau
  (PyTorch pour l'autograd/l'optimiseur, Hugging Face `transformers` pour GPT-2 et
  BERT, `tokenizers`, `datasets` pour télécharger les corpus).
- **Codé par nous** : tout l'**orchestration** — la boucle d'entraînement (precision
  mixte, accumulation de gradient, scheduler, checkpoints, accuracy, sauvegarde sur
  Ctrl-C), le **nettoyage/normalisation des données**, le **pipeline de tokenisation
  en cache memory-mapped**, l'**évaluation**, et **tout le jeu** (logique + interface).
- **From scratch** = aucun poids pré-entraîné externe : on initialise les modèles
  aléatoirement (`GPT2Config` → `GPT2LMHeadModel`) et on entraîne les tokenizers sur
  notre propre corpus. Les `from_pretrained` ne chargent que **nos** checkpoints locaux.

## Les 4 décisions techniques à mettre en avant
1. **Diagnostic des caractères corrompus (`�`)** : le tokenizer ByteLevel encode en
   octets UTF-8 ; un modèle sous-entraîné rate les caractères multi-octets (14 677
   apostrophes courbes dans le corpus). → On **normalise la typographie vers l'ASCII
   en gardant les accents**. Cause traitée à la racine, pas masquée.
2. **Qualité des données > volume** : on a voulu ajouter le corpus Gutenberg, mais un
   filtre quantitatif sur le taux d'accents a montré que **~99 % est du français sans
   accents** → on l'a écarté. On a préféré Wikipédia + FineWeb + Wikisource, propres.
3. **Lois d'échelle (Chinchilla)** : un modèle de 98 M paramètres veut ~2 milliards de
   tokens pour être optimal ; on en a ~620 M sur un laptop → on entraîne donc
   **partiellement** (avec un LR qui décroît proprement sur le budget choisi).
4. **Contrainte matérielle 8 Go VRAM** : precision mixte (AMP) + gradient checkpointing
   + dataset en lecture paresseuse (memmap) pour ne pas saturer la mémoire.

## Chiffres clés
- Corpus final : **~3 Go de texte propre → ~620 M tokens**, 4 sources.
- Mini-GPT : **97,7 M paramètres**, contexte 512, vocabulaire BPE 16 000.
- Métriques suivies : **loss** + **accuracy de prédiction du token suivant**, et
  **perplexité de validation** sur un set tenu à l'écart.

## Limites assumées
Modèle entraîné sous l'optimum Chinchilla (laptop) ; étapes suivantes prévues :
fine-tuning supervisé (SFT) puis optimisation par préférences (DPO) avec le juge de
cohérence comme signal de récompense.
