# Arrete le backend (port 8000) et le frontend (port 5173).
# Usage :  .\stop.ps1

$stopped = 0

# 1) Processus lances par start.ps1 (uvicorn --reload spawn un process enfant, npm/vite aussi).
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'uvicorn backend\.main' -or $_.CommandLine -match 'frontend.*vite' -or $_.CommandLine -match '\\vite\\' } |
    ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force; Write-Host "Arrete pid $($_.ProcessId)" -ForegroundColor Yellow; $stopped++ } catch {}
    }

# 2) Filet de securite : tout ce qui ecoute encore sur les ports du projet.
foreach ($port in 8000, 5173) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object {
            try { Stop-Process -Id $_ -Force; Write-Host "Arrete pid $_ (port $port)" -ForegroundColor Yellow; $stopped++ } catch {}
        }
}

if ($stopped -eq 0) { Write-Host "Aucun serveur en cours." -ForegroundColor Green }
else { Write-Host "$stopped processus arretes." -ForegroundColor Green }
