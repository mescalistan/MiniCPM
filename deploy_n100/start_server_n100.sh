#!/usr/bin/env bash
# ==============================================================================
# Script di avvio per llama-server su Mini PC Intel N100 (Linux / Ubuntu)
# Ottimizzato per 4 E-core Gracemont con istruzioni AVX2 + VNNI e 12 GB RAM
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}

echo "=========================================================="
echo " MiniCPM-2B: llama-server su Intel N100 (Linux / VNNI)"
echo "=========================================================="

# 1. Rileva eseguibile llama-server
SERVER_BIN=""
if command -v llama-server &> /dev/null; then
    SERVER_BIN="llama-server"
elif [ -f "./llama-server" ]; then
    SERVER_BIN="./llama-server"
elif [ -f "../llama_bin/llama-server" ]; then
    SERVER_BIN="../llama_bin/llama-server"
else
    echo "ERRORE: llama-server non trovato! Installa llama.cpp o copia il binario in questa cartella."
    exit 1
fi

# 2. Rileva modello GGUF
MODEL=""
if [ -n "$1" ]; then
    MODEL="$1"
else
    # Cerca il file Q4_K_M o qualsiasi .gguf in . o ./models
    MODEL=$(find . ./models -maxdepth 2 -name "*.gguf" 2>/dev/null | head -n 1)
    if [ -z "$MODEL" ]; then
        echo "ERRORE: Nessun file .gguf trovato in . o ./models!"
        echo "Copia il modello (es. minicpm-2b-q4_k_m.gguf) prima di avviare lo script."
        exit 1
    fi
fi

echo "Modello:    $MODEL"
echo "Porta:      $PORT"
echo "Threads:    4 (Core Fisici E-Cores N100)"
echo "MemLock:    Attivo (Zero paging su swap)"
echo "Endpoint:   http://localhost:$PORT/v1/chat/completions"
echo "=========================================================="
echo ""

# Esegui con i parametri ottimizzati per Intel N100
exec "$SERVER_BIN" \
    -m "$MODEL" \
    -t 4 \
    --threads-batch 4 \
    -c 2048 \
    -b 512 \
    -ub 256 \
    --mlock \
    --min-p 0.0 \
    --jinja \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
