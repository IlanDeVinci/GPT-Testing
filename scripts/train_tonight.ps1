# =============================================================================
#  Entrainement "du soir" - un meilleur mini-GPT from scratch
# =============================================================================
#
#  POURQUOI ce modele plutot que le 97.7M actuel ?
#  Le corpus clean-v3.txt = 621,6 M tokens. La loi de Chinchilla dit qu'un
#  modele est "bien nourri" avec ~20 tokens par parametre, soit un optimum a
#  ~31 M parametres pour ce corpus. Le modele actuel (97,7 M) est 3x trop gros
#  pour la quantite de donnees -> il reste sous-entraine (perplexite/mot ~365).
#  Ici on entraine un ~34 M, dimensionne pour les donnees, sur 2 epoques
#  completes (~1,24 milliard de tokens vus). Mieux nourri = plus coherent.
#
#  MATERIEL : RTX 4060 Laptop (8,6 Go). batch 24 x block 512 + AMP tient en VRAM.
#  Mesure : ~1,4 batch/s avec attention SDPA + TF32 (actives automatiquement).
#  Si "CUDA out of memory" : baisse -batch_size a 16 et monte
#  -gradient_accumulation_steps a 3 (taille de batch effective inchangee = 48).
#
#  BUDGET : -max_steps 14000 (~8-9 h, debit mesure 0,46 update/s ; plus rapide
#  quand le cache disque se rechauffe). Le scheduler cosine decroit le LR jusqu'a
#  ~0 PILE sur ces 14000 pas -> le modele est "fini" proprement au reveil.
#  Cela represente ~344 M tokens vus (~10 tokens/parametre).
#  Un checkpoint est sauvegarde a la fin ET sur Ctrl-C : jamais de travail perdu.
#  Plus de temps (week-end) ? Monte -max_steps a 25000 pour 1 epoque complete.
#
#  LANCER (depuis n'importe ou) :
#      powershell -ExecutionPolicy Bypass -File scripts\train_tonight.ps1
#
#  Le jeu utilisera AUTOMATIQUEMENT checkpoints-v4 des qu'il existe
#  (priorite ajoutee dans backend/paths.py).
# =============================================================================

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv-gpu\Scripts\python.exe"

Write-Host "=== Entrainement du meilleur modele (~34M) - debut ===" -ForegroundColor Cyan
Write-Host "Racine : $root"

& $py scripts\train_model.py `
    --corpus data\clean-v3.txt `
    --tokenizer tokenizer-v3\tokenizer.json `
    --output_dir checkpoints-v4 `
    --sequence_mode continuous `
    --n_positions 512 `
    --block_size 512 `
    --n_layer 8 `
    --n_embd 512 `
    --n_head 8 `
    --epochs 1 `
    --max_steps 14000 `
    --batch_size 24 `
    --gradient_accumulation_steps 2 `
    --learning_rate 6e-4 `
    --weight_decay 0.01 `
    --warmup_ratio 0.03 `
    --num_workers 4 `
    --log_every 50

if ($LASTEXITCODE -ne 0) {
    Write-Host "Entrainement interrompu ou en erreur (code $LASTEXITCODE)." -ForegroundColor Yellow
    Write-Host "Un checkpoint a quand meme pu etre sauvegarde dans checkpoints-v4." -ForegroundColor Yellow
}

# --- Apres l'entrainement : courbe + evaluation comparative ---
Write-Host "`n=== Graphe des courbes d'entrainement ===" -ForegroundColor Cyan
& $py scripts\plot_training.py --metrics checkpoints-v4\metrics.jsonl --out outputs\curve-v4.png

Write-Host "`n=== Evaluation comparee a tous les autres modeles ===" -ForegroundColor Cyan
& $py scripts\eval_all.py --val_corpus data\clean-v3-val.txt

Write-Host "`n=== Termine. Le jeu chargera checkpoints-v4 automatiquement. ===" -ForegroundColor Green
Write-Host "Pour comparer les generations a la main :" -ForegroundColor Green
Write-Host '  uvicorn backend.main:app --reload   puis ouvre le jeu' -ForegroundColor Green
