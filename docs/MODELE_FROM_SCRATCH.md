# Modèle « from scratch » — fiche et sauvegarde

Snapshot du travail entièrement entraîné localement, **sans aucun poids pré-entraîné**.
À conserver tel quel : le futur travail sur un modèle pré-entraîné ira dans des
dossiers **distincts** (`checkpoints-pre-*`) et n'écrasera rien ici.

## Lignage GPT (causal)
| Étape | Dossier | Description |
|---|---|---|
| Base | `checkpoints-v3/` | GPT-2 maison ~98 M params, contexte 512, vocab BPE 16 000, entraîné sur ~620 M tokens |
| SFT | `checkpoints-sft/` (`checkpoint-epoch-1-step-4000`) | fine-tuning narratif (OPUS + Wikisource), loss masquée sur la suite |
| DPO | `checkpoints-dpo/` | préférences notées par le juge BERT |
| Tokenizer | `tokenizer-v3/tokenizer.json` | BPE ByteLevel 16 000, entraîné sur le corpus propre |

## Mini-BERT (fill-mask du jeu)
| Dossier | Description |
|---|---|
| `tinybert/mlm-v3/` | MLM ~9 M params réentraîné sur corpus propre |
| `tinybert/tokenizer-v3/` | WordPiece 16 000 |

## Données
`data/clean-v3.txt` — OPUS Books + Wikipédia FR + FineWeb-2 (fra) + Wikisource,
normalisé (typographie → ASCII, accents conservés), ~620 M tokens.
Validation : `data/clean-v3-val.txt`.

## Métriques (base)
- perplexité validation ≈ **42**, accuracy token-suivant ≈ **33 %**
- caractères corrompus ≈ **0** (après normalisation typographique)

## Reproduire
Voir [FONCTIONNEMENT.md](FONCTIONNEMENT.md) et [AMELIORATION_GPT.md](AMELIORATION_GPT.md).
Pipeline : `download_french_corpus` → `clean_data` → `train_tokenizer` →
`train_model` → `train_sft` → `build_dpo_data` → `train_dpo`.

## Limite assumée
Un modèle de 98 M *from scratch* sur laptop 8 Go plafonne (~perplexité 42). Pour aller
plus loin, on pivote vers un **modèle français pré-entraîné** + SFT/DPO (lignage séparé).
