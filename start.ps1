# Lance le backend (FastAPI) et le frontend (Vite) dans deux fenetres, puis ouvre le jeu.
# Usage :  .\start.ps1     (ou clic droit > Executer avec PowerShell)

$root = $PSScriptRoot

# Premiere fois seulement : installe les dependances frontend si besoin.
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "Installation des dependances frontend..." -ForegroundColor Cyan
    Push-Location "$root\frontend"; npm install; Pop-Location
}

Write-Host "Demarrage du backend (port 8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; .\.venv-gpu\Scripts\Activate.ps1; uvicorn backend.main:app --reload"
)

Write-Host "Demarrage du frontend (port 5173)..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"
Write-Host "Jeu disponible sur http://localhost:5173 (le 1er chargement des modeles prend ~20s)." -ForegroundColor Yellow
