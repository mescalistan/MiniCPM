# PowerShell Launcher per llama-server su Mini PC Intel N100 (Windows)
# Flag ottimizzati per 4 E-core Gracemont, AVX2 + VNNI e 12 GB RAM.

param(
    [string]$Model = "",
    [int]$Port = 8080,
    [string]$HostIP = "0.0.0.0"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " MiniCPM-2B: llama-server su Intel N100 (4 Threads / VNNI)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Rileva eseguibile llama-server.exe
$ServerExe = "llama-server.exe"
if (-not (Get-Command $ServerExe -ErrorAction SilentlyContinue)) {
    if (Test-Path ".\llama-server.exe") {
        $ServerExe = ".\llama-server.exe"
    } elseif (Test-Path "..\llama_bin\llama-server.exe") {
        $ServerExe = "..\llama_bin\llama-server.exe"
    } else {
        Write-Error "llama-server.exe non trovato! Assicurati di copiare i binari nella cartella o nel PATH."
        exit 1
    }
}

# 2. Rileva modello GGUF
if (-not $Model) {
    $candidates = Get-ChildItem -Path ".", ".\models" -Filter "*.gguf" -ErrorAction SilentlyContinue | Sort-Object Length
    if ($candidates.Count -gt 0) {
        $Model = $candidates[0].FullName
    } else {
        Write-Error "Nessun file .gguf trovato in . o ./models! Copia prima il file minicpm-2b-q4_k_m.gguf."
        exit 1
    }
}

Write-Host "Modello:    $Model" -ForegroundColor Green
Write-Host "Porta:      $Port" -ForegroundColor Green
Write-Host "Threads:    4 (Core Fisici E-Cores N100)" -ForegroundColor Green
Write-Host "MemLock:    Attivo (Zero paging su disco)" -ForegroundColor Green
Write-Host "Endpoint:   http://localhost:$Port/v1/chat/completions" -ForegroundColor Yellow
Write-Host "==========================================================`n" -ForegroundColor Cyan

& $ServerExe `
    -m $Model `
    -t 4 `
    --threads-batch 4 `
    -c 2048 `
    -b 512 `
    -ub 256 `
    --mlock `
    --min-p 0.0 `
    --jinja `
    --host $HostIP `
    --port $Port
