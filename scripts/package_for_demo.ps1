# =============================================================================
#  Prepare demo-package/ + demo-package.zip avec UNIQUEMENT ce qu'il faut pour
#  FAIRE TOURNER le jeu sur un autre PC (pas pour reentrainer).
#
#  Inclut : code (backend + frontend), checkpoints FINAUX (sans l'optimiseur
#  training_state.pt ni les epochs intermediaires), tokenizers, requirements.
#  Option -IncludeRag : ajoute data/rag-index (~123 Mo) pour le filet RAG.
#  Exclut : corpora data/ (4 Go) et le venv (a RECREER sur le laptop).
#
#  LANCER :
#    powershell -ExecutionPolicy Bypass -File scripts\package_for_demo.ps1
#    powershell -ExecutionPolicy Bypass -File scripts\package_for_demo.ps1 -IncludeRag
# =============================================================================

param(
    [switch]$IncludeRag,
    [switch]$IncludeV4,        # ajoute aussi le GPT base v4 (sinon narratif suffit pour le jeu)
    [string]$Dest = "demo-package"
)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$dst = Join-Path $root $Dest
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dst | Out-Null

function Copy-Dir($rel, $exclude = @()) {
    $src = Join-Path $root $rel
    if (-not (Test-Path $src)) { Write-Host "  (absent) $rel" -ForegroundColor DarkGray; return }
    $target = Join-Path $dst $rel
    $args = @($src, $target, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
    if ($exclude.Count) { $args += "/XF"; $args += $exclude }
    robocopy @args | Out-Null
    Write-Host "  + $rel\" -ForegroundColor Gray
}

function Copy-File($rel) {
    $src = Join-Path $root $rel
    if (Test-Path $src) {
        $t = Join-Path $dst $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $t -Parent) | Out-Null
        Copy-Item $src $t -Force
        Write-Host "  + $rel"
    }
}

# Copie un dossier de checkpoints : SEULEMENT le checkpoint final (latest.json)
# et SANS l'etat de l'optimiseur (training_state.pt).
function Copy-Checkpoint($ckptDir) {
    $latest = Join-Path $root "$ckptDir\latest.json"
    if (-not (Test-Path $latest)) { Write-Host "  (absent) $ckptDir" -ForegroundColor DarkGray; return }
    $name = (Get-Content $latest -Raw | ConvertFrom-Json).checkpoint
    Copy-File "$ckptDir\latest.json"
    Copy-Dir "$ckptDir\$name" @("training_state.pt")
}

Write-Host "Preparation de $dst ..." -ForegroundColor Cyan

# --- Code & config ---
Copy-Dir "backend"
Copy-Dir "frontend"
Copy-File "requirements.txt"
Copy-File "README.md"

# --- Tokenizers ---
Copy-Dir "tokenizer-v3"
Copy-Dir "tinybert\tokenizer-v3"

# --- Checkpoints (finaux, sans optimiseur) ---
Copy-Checkpoint "checkpoints-narratif"      # GPT du jeu (defaut)
if ($IncludeV4) { Copy-Checkpoint "checkpoints-v4" }
Copy-Checkpoint "tinybert\mlm-v3"           # BERT (remplissage)

# --- RAG (optionnel) ---
if ($IncludeRag) { Copy-Dir "data\rag-index" }

# --- Zip ---
$zip = "$dst.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$dst\*" -DestinationPath $zip -CompressionLevel Optimal

$sizeMB = [math]::Round((Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 0)
$zipMB = [math]::Round((Get-Item $zip).Length / 1MB, 0)
Write-Host "`nPret : $Dest\  (~$sizeMB Mo)  ->  $Dest.zip (~$zipMB Mo)" -ForegroundColor Green
Write-Host "Sur le laptop : dezippe, puis dans le dossier :" -ForegroundColor Green
Write-Host "  python -m venv .venv ; .venv\Scripts\activate ; pip install -r requirements.txt" -ForegroundColor Green
Write-Host "  uvicorn backend.main:app --reload    (CPU si pas de GPU NVIDIA, plus lent mais OK)" -ForegroundColor Green
