# Améliorer le mini-GPT — diagnostic et choix

Note destinée à expliquer *pourquoi* ces choix, pas seulement *quoi* faire.

## 1. Diagnostic du problème (`�` dans les générations)

Le tokenizer du GPT est un **BPE ByteLevel** : il encode chaque caractère en octets
UTF-8, donc il ne produit jamais de `[UNK]`. Les `�` n'apparaissent qu'à la
**génération** : un modèle sous-entraîné émet une séquence d'octets qui ne forme pas
un caractère UTF-8 valide, et le décodeur la remplace par `�` (U+FFFD).

Plus un caractère est **multi-octets et rare**, plus il est fragile à reproduire :

| caractère            | octets UTF-8 | occurrences dans OPUS |
|----------------------|--------------|-----------------------|
| `'` apostrophe courbe | 3 (`E2 80 99`) | **14 677**           |
| `«` / `»` guillemets  | 2 chacun       | 2 206 / 1 681        |
| `…` points de suspension | 3           | 661                  |
| `—` tiret cadratin    | 3             | 119                  |

Le GPT actif a été entraîné sur ce texte non normalisé, avec seulement
**2 540 mises à jour** sur ~685 k mots : trop peu pour mémoriser 14 k+ séquences
d'octets fragiles. D'où les `d'un`, `l'air` cassés.

## 2. Choix n°1 — normaliser la typographie, garder les accents

On remplace la ponctuation typographique par ses équivalents ASCII **1 octet**
(`'→'`, `«»→"`, `…→...`, `—→-`) dans `scripts/clean_data.py`, tout en **conservant
les accents** (`é è à ç …`), qui sont fréquents et donc réellement appris.

Résultat sur le corpus régénéré (`data/clean-v3.txt`) :

- apostrophes courbes : `14 677 → 0`
- accents préservés : 52 453 `é` toujours présents

Pourquoi pas supprimer aussi les accents ? Parce qu'ils portent le sens en français
(`a`/`à`, `ou`/`où`) et sont assez fréquents pour être appris correctement. On retire
seulement le bruit multi-octets *inutile*, pas l'information linguistique.

## 3. Choix n°2 — ne PAS ajouter le corpus Gutenberg

Tentation : Gutenberg est ~15× plus gros (10,7 M mots). Mais un filtre quantitatif
sur le taux de lettres accentuées (`--min_accent_ratio 0.012`) rejette
**201 644 lignes sur 223 553**, soit ~99 % de Gutenberg : c'est du français ancien
sans accents (`etait` au lieu de `était`). L'ajouter apprendrait une mauvaise
orthographe.

Leçon défendable : **la qualité des données prime sur le volume**. On préfère un
petit corpus propre (OPUS Books normalisé) à un gros corpus dégradé.

## 4. Choix n°3 — entraîner plus longtemps, puis chercher plus de données propres

Une fois les données propres, le levier suivant est le **volume d'entraînement**,
pas une technique d'alignement. On réentraîne from scratch (le vocabulaire change) :

```powershell
# 1. corpus normalisé  (déjà généré -> data/clean-v3.txt)
python scripts/clean_data.py --inputs data/raw.txt --output data/clean-v3.txt

# 2. tokenizer          (déjà généré -> tokenizer-v3/)
python scripts/train_tokenizer.py --corpus data/clean-v3.txt --output tokenizer-v3/tokenizer.json --vocab_size 8000

# 3. entraînement plus long
python scripts/train_model.py --corpus data/clean-v3.txt --tokenizer tokenizer-v3/tokenizer.json --output_dir checkpoints-v3 --epochs 20 --batch_size 16 --block_size 256 --n_positions 256 --n_embd 256 --n_layer 4 --n_head 4 --learning_rate 5e-4
```

Pour activer ce modèle dans le jeu, pointer `backend/paths.py` vers `checkpoints-v3`
et `tokenizer-v3/tokenizer.json`.

Limite honnête : 685 k mots reste petit. La normalisation supprime les `�`, mais pour
gagner en **fluidité** il faut plus de français propre — par ex. davantage de paires
OPUS Books, ou Wikipédia FR / OSCAR-fr filtrés avec le même critère d'accents.

## 5. Pourquoi pas du RLHF tout de suite ?

L'ordre des leviers par retour sur investissement :

1. données propres + assez d'entraînement  ← on est ici
2. SFT (fine-tuning supervisé sur le style)
3. DPO (préférences, sans boucle RL ; le juge de cohérence archivé peut labelliser)
4. reward model + PPO/GRPO (RLHF complet)

Aligner (SFT/RLHF) un modèle de base sous-entraîné revient à polir une fondation
fissurée : les gains sont marginaux et instables. On corrige d'abord la base.

## 6. Comment mesurer l'amélioration

- **Perplexité** sur un petit ensemble de validation tenu à l'écart (held-out).
- **Taux de `�`** par 1 000 tokens générés (doit tomber à ~0 après normalisation).
- **Lisibilité** : lecture humaine de N générations à température fixe, avant/après.
