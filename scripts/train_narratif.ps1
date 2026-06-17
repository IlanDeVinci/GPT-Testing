# =============================================================================
#  Specialisation narrative - rendre le from-scratch MEILLEUR que le
#  pre-entraine SUR SON DOMAINE (les histoires).
# =============================================================================
#
#  IDEE. On ne peut pas battre asi/gpt-fr-cased-small sur le francais general
#  (il a vu ~100x plus de donnees). Mais un petit modele dont TOUTE la capacite
#  est concentree sur le narratif peut le battre SUR LE NARRATIF.
#
#  METHODE (= adaptation de domaine, PAS le SFT agressif qui avait tout casse) :
#  on reprend les POIDS de la base from-scratch (v4 si entraine cette nuit, sinon
#  v3), et on continue l'entrainement en LM causal sur le corpus narratif avec un
#  LR FAIBLE (3e-5) + optimiseur neuf. Faible LR = adaptation sans oubli brutal.
#  On evalue ensuite sur data/narratif-val.txt (tenu a l'ecart) : c'est LA le
#  resultat a montrer.
#
#  PREREQUIS : le corpus a deja ete construit par
#    python scripts/clean_data.py --inputs data/clean-narratif.txt data/wikisource-fr.txt data/clean.txt `
#        --output data/narratif-train.txt --val_output data/narratif-val.txt --val_every 50 --min_chars 40
#
#  DUREE : ~1 h (15 M tokens x 3 epoques). Checkpoint sauvegarde a chaque epoque.
#
#  LANCER (idealement APRES train_tonight.ps1 pour partir de v4) :
#    powershell -ExecutionPolicy Bypass -File scripts\train_narratif.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv-gpu\Scripts\python.exe"

# Base de depart : v4 en priorite (mieux entraine), sinon v3.
if (Test-Path (Join-Path $root "checkpoints-v4\latest.json")) {
    $baseDir = "checkpoints-v4"
} elseif (Test-Path (Join-Path $root "checkpoints-v3\latest.json")) {
    $baseDir = "checkpoints-v3"
} else {
    Write-Host "Aucune base from-scratch (checkpoints-v4 ou v3). Lance d'abord train_tonight.ps1." -ForegroundColor Red
    exit 1
}
$base = (& $py -c "import json,sys; d=sys.argv[1]; print(d + '\\' + json.load(open(d+'\\latest.json'))['checkpoint'])" (Join-Path $root $baseDir))
Write-Host "=== Specialisation narrative depuis $base ===" -ForegroundColor Cyan

& $py scripts\train_model.py `
    --corpus data\narratif-train.txt `
    --tokenizer tokenizer-v3\tokenizer.json `
    --output_dir checkpoints-narratif `
    --resume_checkpoint $base `
    --fresh_optimizer `
    --sequence_mode continuous `
    --n_positions 512 `
    --block_size 512 `
    --epochs 3 `
    --batch_size 24 `
    --gradient_accumulation_steps 2 `
    --learning_rate 3e-5 `
    --weight_decay 0.01 `
    --warmup_ratio 0.05 `
    --num_workers 4 `
    --log_every 25

if ($LASTEXITCODE -ne 0) {
    Write-Host "Interrompu/erreur (code $LASTEXITCODE) - un checkpoint a pu etre sauvegarde." -ForegroundColor Yellow
}

Write-Host "`n=== Courbe d'entrainement ===" -ForegroundColor Cyan
& $py scripts\plot_training.py --metrics checkpoints-narratif\metrics.jsonl --out outputs\curve-narratif.png

Write-Host "`n=== LE test qui compte : perplexite SUR LE NARRATIF (held-out) ===" -ForegroundColor Cyan
Write-Host "Compare le from-scratch specialise au pre-entraine sur data/narratif-val.txt." -ForegroundColor Cyan
& $py scripts\eval_all.py --val_corpus data\narratif-val.txt --out_json outputs\eval_narratif.json --out_png outputs\eval_narratif.png

Write-Host "`n=== Termine. Si le from-scratch specialise a la perplexite/mot la plus basse" -ForegroundColor Green
Write-Host "    sur narratif-val, tu as gagne sur ton domaine. Le jeu l'utilise automatiquement." -ForegroundColor Green
