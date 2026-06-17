# =============================================================================
#  PIPELINE COMPLET - entraine tout, avec graphes EN DIRECT
# =============================================================================
#
#  CE QUE FAIT CE SCRIPT, dans l'ordre :
#
#   Phase 1  GPT base from-scratch (~34M, Chinchilla-optimal)   -> checkpoints-v4
#   Phase 2  Adaptation de domaine narrative (le "fine-tune"     -> checkpoints-narratif
#            qui AIDE : LR faible, depuis la base, sans oubli)
#   Phase 3  (option -RetrainBert) mini-BERT MLM from-scratch    -> tinybert/mlm-v3
#   Final    Evaluation comparative (general + narratif) + graphes
#
#  A chaque phase, une 2e fenetre s'ouvre avec la COURBE EN DIRECT (loss +
#  accuracy, rafraichie toutes les 3 s). Elle se ferme a la fin de la phase.
#
#  POURQUOI PAS DE SFT / DPO classiques ?
#  Ton evaluation (outputs/eval_all.png) a MESURE qu'ils degradent le from-scratch
#  (perplexite/mot 365 -> 842 -> 880). On ne les rejoue donc pas : l'adaptation
#  narrative de la Phase 2 est le fine-tuning qui marche. Les anciens checkpoints
#  sft/dpo restent pour la comparaison dans le rendu.
#
#  Le mini-BERT est DEJA entraine (tinybert/mlm-v3). Ne le reentraine (-RetrainBert)
#  que si tu veux refaire la demonstration ; sinon la Phase 3 est sautee.
#
#  LANCER :
#    powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1
#    powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 -RetrainBert
#    powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 -SkipBase   # si v4 deja fait
# =============================================================================

param(
    [switch]$SkipBase,        # saute la Phase 1 si checkpoints-v4 existe deja
    [switch]$SkipNarratif,    # saute la Phase 2
    [switch]$RetrainBert      # active la Phase 3 (sinon sautee : BERT deja entraine)
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv-gpu\Scripts\python.exe"

# Ouvre une fenetre de graphe en direct sur un fichier metrics.jsonl, renvoie le process.
function Start-LiveGraph($metricsRelPath) {
    return Start-Process -FilePath $py `
        -ArgumentList "scripts\plot_training.py", "--live", "--metrics", $metricsRelPath, "--refresh", "3" `
        -PassThru
}
function Stop-LiveGraph($proc) {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue }
}

# --- Phase 1 : GPT base from-scratch -------------------------------------------------
if (-not $SkipBase) {
    Write-Host "`n========== PHASE 1/3 : GPT base from-scratch (~34M) ==========" -ForegroundColor Cyan
    Write-Host "Graphe en direct dans une 2e fenetre. Duree ~8-9 h." -ForegroundColor Cyan
    $g = Start-LiveGraph "checkpoints-v4\metrics.jsonl"
    try {
        & $py scripts\train_model.py `
            --corpus data\clean-v3.txt --tokenizer tokenizer-v3\tokenizer.json `
            --output_dir checkpoints-v4 --sequence_mode continuous `
            --n_positions 512 --block_size 512 --n_layer 8 --n_embd 512 --n_head 8 `
            --epochs 1 --max_steps 14000 --batch_size 24 --gradient_accumulation_steps 2 `
            --learning_rate 6e-4 --weight_decay 0.01 --warmup_ratio 0.03 `
            --num_workers 4 --log_every 50
    } finally { Stop-LiveGraph $g }
    & $py scripts\plot_training.py --metrics checkpoints-v4\metrics.jsonl --out outputs\curve-v4.png
} else {
    Write-Host "`n[Phase 1 sautee] checkpoints-v4 conserve." -ForegroundColor DarkGray
}

# --- Phase 2 : adaptation de domaine narrative ---------------------------------------
if (-not $SkipNarratif) {
    Write-Host "`n========== PHASE 2/3 : adaptation narrative (fine-tune utile) ==========" -ForegroundColor Cyan
    # Base = v4 si dispo, sinon v3.
    if (Test-Path (Join-Path $root "checkpoints-v4\latest.json")) { $baseDir = "checkpoints-v4" }
    elseif (Test-Path (Join-Path $root "checkpoints-v3\latest.json")) { $baseDir = "checkpoints-v3" }
    else { Write-Host "Aucune base (v4/v3). Phase 2 sautee." -ForegroundColor Red; $baseDir = $null }

    if ($baseDir) {
        if (-not (Test-Path (Join-Path $root "data\narratif-train.txt"))) {
            Write-Host "Construction du corpus narratif (train + val held-out)..." -ForegroundColor Cyan
            & $py scripts\clean_data.py --inputs data\clean-narratif.txt data\wikisource-fr.txt data\clean.txt `
                --output data\narratif-train.txt --val_output data\narratif-val.txt --val_every 50 --min_chars 40
        }
        $base = (& $py -c "import json,sys; d=sys.argv[1]; print(d + '\\' + json.load(open(d+'\\latest.json'))['checkpoint'])" (Join-Path $root $baseDir))
        Write-Host "Depuis $base . Graphe en direct dans une 2e fenetre. Duree ~1 h." -ForegroundColor Cyan
        $g = Start-LiveGraph "checkpoints-narratif\metrics.jsonl"
        try {
            & $py scripts\train_model.py `
                --corpus data\narratif-train.txt --tokenizer tokenizer-v3\tokenizer.json `
                --output_dir checkpoints-narratif --resume_checkpoint $base --fresh_optimizer `
                --sequence_mode continuous --n_positions 512 --block_size 512 `
                --epochs 3 --batch_size 24 --gradient_accumulation_steps 2 `
                --learning_rate 3e-5 --weight_decay 0.01 --warmup_ratio 0.05 `
                --num_workers 4 --log_every 25
        } finally { Stop-LiveGraph $g }
        & $py scripts\plot_training.py --metrics checkpoints-narratif\metrics.jsonl --out outputs\curve-narratif.png
    }
} else {
    Write-Host "`n[Phase 2 sautee]." -ForegroundColor DarkGray
}

# --- Phase 3 (option) : mini-BERT MLM ------------------------------------------------
if ($RetrainBert) {
    Write-Host "`n========== PHASE 3/3 : mini-BERT MLM from-scratch ==========" -ForegroundColor Cyan
    Write-Host "Graphe en direct dans une 2e fenetre." -ForegroundColor Cyan
    $g = Start-LiveGraph "tinybert\mlm-v3\metrics.jsonl"
    try {
        & $py scripts\train_tinybert_mlm.py `
            --corpus data\clean-bert.txt --tokenizer_dir tinybert\tokenizer-v3 `
            --output_dir tinybert\mlm-v3 --max_length 256 --hidden_size 256 `
            --num_hidden_layers 6 --num_attention_heads 4 --intermediate_size 1024 `
            --epochs 3 --batch_size 32 --learning_rate 5e-4 --log_every 50
    } finally { Stop-LiveGraph $g }
    & $py scripts\plot_training.py --metrics tinybert\mlm-v3\metrics.jsonl --out outputs\curve-bert.png
} else {
    Write-Host "`n[Phase 3 sautee] mini-BERT deja entraine (tinybert/mlm-v3)." -ForegroundColor DarkGray
}

# --- Index RAG (recuperation sur le mini-BERT) ---------------------------------------
if (Test-Path (Join-Path $root "data\narratif-train.txt")) {
    Write-Host "`n========== INDEX RAG (mini-BERT comme encodeur) ==========" -ForegroundColor Cyan
    & $py scripts\build_rag_index.py --corpus data\narratif-train.txt --max_passages 200000
} else {
    Write-Host "`n[Index RAG saute] corpus narratif absent." -ForegroundColor DarkGray
}

# --- Final : evaluation comparative --------------------------------------------------
Write-Host "`n========== EVALUATION FINALE ==========" -ForegroundColor Green
Write-Host "Sur le francais general :" -ForegroundColor Green
& $py scripts\eval_all.py --val_corpus data\clean-v3-val.txt --out_json outputs\eval_all.json --out_png outputs\eval_all.png
Write-Host "`nSur le NARRATIF (le test qui compte pour ton domaine) :" -ForegroundColor Green
& $py scripts\eval_all.py --val_corpus data\narratif-val.txt --out_json outputs\eval_narratif.json --out_png outputs\eval_narratif.png

Write-Host "`n=== Pipeline termine. Graphes dans outputs/. Le jeu charge automatiquement" -ForegroundColor Green
Write-Host "    le meilleur modele (checkpoints-narratif). ===" -ForegroundColor Green
