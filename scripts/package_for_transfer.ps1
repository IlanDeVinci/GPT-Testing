# =============================================================================
#  Prepare un dossier transfer/ avec UNIQUEMENT ce qu'il faut pour entrainer
#  sur un autre PC. Voir docs/MIGRATION.md.
#
#  Par defaut (~2,8 Go) : corpus nettoye + tokenizers + mini-BERT + sorties.
#  -IncludeCheckpoints : ajoute aussi les checkpoints pre-entraines (~1,9 Go).
#
#  LANCER :
#    powershell -ExecutionPolicy Bypass -File scripts\package_for_transfer.ps1
#    powershell -ExecutionPolicy Bypass -File scripts\package_for_transfer.ps1 -IncludeCheckpoints
# =============================================================================

param(
    [switch]$IncludeCheckpoints,
    [string]$Dest = "transfer"
)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$dst = Join-Path $root $Dest

# Fichiers de donnees essentiels (le corpus nettoye, pas les bruts ni les caches).
$dataFiles = @(
    "data\clean-v3.txt",
    "data\clean-bert.txt",
    "data\clean-v3-val.txt",
    "data\narratif-train.txt",
    "data\narratif-val.txt",
    "data\corpus_info.json"
)
# Dossiers a copier entierement.
$dirs = @(
    "tokenizer-v3",
    "tinybert\tokenizer-v3",
    "tinybert\mlm-v3",
    "outputs"
)
if ($IncludeCheckpoints) {
    $dirs += "checkpoints-pre-sft"
    $dirs += "checkpoints-pre-dpo"
}

Write-Host "Preparation de $dst ..." -ForegroundColor Cyan

# Copie des fichiers (en preservant data/).
New-Item -ItemType Directory -Force -Path (Join-Path $dst "data") | Out-Null
foreach ($f in $dataFiles) {
    $src = Join-Path $root $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $dst $f) -Force
        Write-Host "  + $f"
    } else {
        Write-Host "  (absent, ignore) $f" -ForegroundColor DarkGray
    }
}

# Copie des dossiers via robocopy (exit code < 8 = succes).
foreach ($d in $dirs) {
    $src = Join-Path $root $d
    if (Test-Path $src) {
        $target = Join-Path $dst $d
        robocopy $src $target /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Write-Host "  ECHEC robocopy : $d" -ForegroundColor Red }
        else { Write-Host "  + $d\" }
    } else {
        Write-Host "  (absent, ignore) $d" -ForegroundColor DarkGray
    }
}
$global:LASTEXITCODE = 0  # robocopy laisse un code non nul meme en succes

# Taille totale.
$sizeMB = [math]::Round((Get-ChildItem $dst -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB, 0)
Write-Host "`nPret : $dst  (~$sizeMB Mo)" -ForegroundColor Green
Write-Host "Copie ce dossier sur le nouveau PC, puis verse son contenu DANS le repo clone." -ForegroundColor Green
Write-Host "Ensuite : voir docs/MIGRATION.md (etapes 3 et 4)." -ForegroundColor Green
