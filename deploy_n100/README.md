# Bundle di Deployment MiniCPM-2B per Mini PC Intel N100

Questa cartella contiene tutto il necessario per eseguire il modello **MiniCPM-2B** quantizzato (`Q4_K_M` o `Q5_K_M`) su un Mini PC con processore **Intel N100 (4 core Gracemont, 12 GB RAM)**.

---

## 1. Struttura dei File da trasferire sul Mini PC

Copia questa cartella sul tuo Mini PC insieme ai binari di `llama.cpp` e al modello `.gguf`:

```
deploy_n100/
├── models/
│   └── minicpm-2b-q4_k_m.gguf   # Pesi del modello quantizzati esportati
├── llama-server.exe             # (Su Windows) Eseguibile di llama-server
├── start_server_n100.ps1        # Script di avvio per Windows
├── start_server_n100.sh         # Script di avvio per Linux (Ubuntu/Debian)
├── docker-compose.yml           # Configurazione opzionale Docker
└── README.md
```

---

## 2. Modalità A: Esecuzione Nativa su Windows

1. Posiziona il file `minicpm-2b-q4_k_m.gguf` nella sottocartella `models/`.
2. Posiziona `llama-server.exe` (e le relative DLL come `ggml-cpu-alderlake.dll`) nella cartella principale.
3. Apri PowerShell ed esegui:
   ```powershell
   .\start_server_n100.ps1
   ```

---

## 3. Modalità B: Esecuzione Nativa su Linux (Ubuntu / Debian)

1. Rendi eseguibile lo script:
   ```bash
   chmod +x start_server_n100.sh
   ```
2. Installa o compila `llama.cpp` (`sudo apt install llama.cpp` o scarica la release precompilata Linux).
3. Avvia il server:
   ```bash
   ./start_server_n100.sh
   ```

---

## 4. Modalità C: Esecuzione Containerizzata con Docker Compose

Se preferisci isolare il processo senza installare librerie di sistema:

```bash
# Avvia il container in background
docker compose up -d

# Visualizza i log in tempo reale
docker compose logs -f

# Arresta il container
docker compose down
```

> [!NOTE]
> Il file `docker-compose.yml` include `ulimits: memlock: soft: -1, hard: -1` per consentire al container di bloccare la memoria del modello nella RAM fisica (`--mlock`) senza privilege escalation.

---

## 5. Come Effettuare Richieste API (OpenAI-Compatible)

Il server espone l'endpoint standard OpenAI `/v1/chat/completions` sulla porta **8080**:

### Con cURL (Terminale):
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Sei un assistente AI conciso e tecnico."},
      {"role": "user", "content": "Spiegami come funziona il garbage collector in Python."}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512
  }'
```

### In Python (usando la libreria `openai` standard):
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<IP_DEL_MINI_PC>:8080/v1",
    api_key="non-necessaria"
)

response = client.chat.completions.create(
    model="minicpm-2b",
    messages=[
        {"role": "system", "content": "Sei un assistente tecnico diretto."},
        {"role": "user", "content": "Scrivi una funzione Python per calcolare la distanza di Levenshtein."}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```
