# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Script automatizzato per il download dei binari ufficiali di llama.cpp,
    la conversione da Hugging Face in GGUF FP16 e la quantizzazione (Q4_K_M / Q5_K_M) per Intel N100.
#>

param(
    [string]$ModelPath = "output/MiniCPM-2B-Merged-FP16",
    [string]$OutputDir = "output/GGUF",
    [string[]]$QuantTypes = @("Q4_K_M", "Q5_K_M"),
    [switch]$KeepF16 = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " MiniCPM-2B: Pipeline Esportazione e Quantizzazione GGUF" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verifica ambiente Python
$PythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# 2. Clona il repository llama.cpp per convert_hf_to_gguf.py
if (-not (Test-Path "llama.cpp\convert_hf_to_gguf.py")) {
    Write-Host "`n[1/5] Clonazione repository llama.cpp (shallow clone)..." -ForegroundColor Yellow
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
} else {
    Write-Host "`n[1/5] Repository llama.cpp gia presente." -ForegroundColor Green
}

# 3. Assicura dipendenze Python per la conversione GGUF
Write-Host "`n[2/5] Verifica dipendenze python (gguf)..." -ForegroundColor Yellow
& $PythonExe -m pip install gguf -q

# 4. Download binari precompilati Windows (AVX2 / Alder Lake-N)
$BinDir = "llama_bin"
$QuantizeExe = Join-Path $BinDir "llama-quantize.exe"

if (-not (Test-Path $QuantizeExe)) {
    Write-Host "`n[3/5] Download binari precompilati ufficiali llama.cpp per Windows..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    
    # Rilevamento ultima release o fallback
    $Tag = "b11818"
    try {
        $resp = Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases" -UseBasicParsing
        $m = [regex]::Match($resp.Content, "/releases/tag/(b\d+)")
        if ($m.Success) {
            $Tag = $m.Groups[1].Value
        }
    } catch {
        Write-Host "Uso tag release di default: $Tag" -ForegroundColor Gray
    }

    $ZipName = "llama-$Tag-bin-win-cpu-x64.zip"
    $DownloadUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/$ZipName"
    $ZipDest = Join-Path $env:TEMP $ZipName

    Write-Host "Scaricamento binari da: $DownloadUrl ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipDest -UseBasicParsing
    
    Write-Host "Estrazione binari in $BinDir ..." -ForegroundColor Cyan
    Expand-Archive -Path $ZipDest -DestinationPath $BinDir -Force
    Remove-Item -Path $ZipDest -Force
    Write-Host "Binari scaricati ed estratti con successo!" -ForegroundColor Green
} else {
    Write-Host "`n[3/5] Binari precompilati llama.cpp gia presenti in $BinDir." -ForegroundColor Green
}

# 5. Verifica modello di input
if (-not (Test-Path $ModelPath)) {
    Write-Host "ATTENZIONE: Percorso $ModelPath non trovato." -ForegroundColor Yellow
    Write-Host "Utilizzo fallback sul checkpoint pre-convertito di Hugging Face (llama-format)..." -ForegroundColor Yellow
    $ModelPath = "openbmb/MiniCPM-2B-dpo-fp16"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$F16Gguf = Join-Path $OutputDir "minicpm-2b-f16.gguf"

# 6. Conversione in GGUF FP16
Write-Host "`n[4/5] Conversione Hugging Face -> GGUF FP16 ($F16Gguf)..." -ForegroundColor Yellow
& $PythonExe llama.cpp/convert_hf_to_gguf.py $ModelPath --outfile $F16Gguf --outtype f16

if (-not (Test-Path $F16Gguf)) {
    Write-Error "Conversione GGUF FP16 fallita!"
    exit 1
}

# 7. Quantizzazione nei formati richiesti (Q4_K_M, Q5_K_M)
Write-Host "`n[5/5] Esecuzione quantizzazioni per Intel N100..." -ForegroundColor Yellow

$GeneratedFiles = @()

foreach ($q in $QuantTypes) {
    $outFile = Join-Path $OutputDir ("minicpm-2b-" + $q.ToLower() + ".gguf")
    Write-Host "--> Quantizzazione in $q -> $outFile ..." -ForegroundColor Cyan
    & $QuantizeExe $F16Gguf $outFile $q
    if (Test-Path $outFile) {
        $GeneratedFiles += $outFile
    }
}

# Pulizia facoltativa del file FP16 da ~5GB
if (-not $KeepF16) {
    Write-Host "`nRimozione file intermedio FP16 per risparmiare 5GB di disco..." -ForegroundColor Gray
    Remove-Item -Path $F16Gguf -Force
}

Write-Host "`n" + ("=" * 60) -ForegroundColor Green
Write-Host "SUCCESS: Quantizzazione GGUF completata!" -ForegroundColor Green
foreach ($file in $GeneratedFiles) {
    $item = Get-Item $file
    $sizeMB = [math]::Round($item.Length / 1MB, 2)
    Write-Host ("  File: {0} ({1} MB)" -f $item.Name, $sizeMB) -ForegroundColor White
}
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "I file sono pronti per essere copiati nella cartella deploy_n100/ !" -ForegroundColor Yellow
