# Script 1-Click per avviare l'interfaccia grafica Web (Gradio)
# Apre automaticamente il browser su http://127.0.0.1:7860

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Ambiente virtuale .venv non trovato!"
    exit 1
}

# Attiva l'ambiente virtuale
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Avvio Interfaccia Web MiniCPM-2B..." -ForegroundColor Cyan
python webui.py $args
