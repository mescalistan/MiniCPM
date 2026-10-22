# Script 1-Click per avviare la chat interattiva nel terminale
# Attiva automaticamente l'ambiente virtuale ed esegue MiniCPM-2B in FP16 su RTX 2080

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Ambiente virtuale .venv non trovato! Esegui prima il setup."
    exit 1
}

# Attiva l'ambiente virtuale
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Avvio MiniCPM-2B in corso..." -ForegroundColor Cyan
python chat_cli.py $args
