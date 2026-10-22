# C-3PO su HOME-SERVER: diagnosi e moduli

Moduli Node.js per `ai-soc-app` (HOME-SERVER) e configurazione di `llama-server` per MiniCPM-2B su Intel N100.
Il codice di HOME-SERVER non sta in questo repository: i file in `home-server/ai-soc-app/lib/` si installano sul
server con `home-server/deploy/deploy_home_server.py`.

## Diagnosi

### 1. Gibberish e loop di caratteri (`ogeoge`, `relúúojoj`)

| Causa | Evidenza | Correzione |
|---|---|---|
| `presence_penalty: 0.3` + `frequency_penalty: 0.3` su ogni richiesta | `lib/ai-engine-client.js` (versione sul server) | Penalità a 0; `repeat_penalty 1.05` + `min_p 0.05` per profilo |
| Il vocabolario di MiniCPM codifica molte lettere accentate come token-byte (`<0xC3>`, `<0xA8>`): le penalità additive li colpiscono dopo pochi usi e il campionatore sceglie byte "vicini", rompendo l'UTF-8 | — | come sopra; in più il client interrompe lo stream se la coda degenera (`isDegenerateTail`) |
| `--cache-ram` vale **8192 MiB** di default nelle build recenti, con `mem_limit: 3500m` | `llama-server --help` (build 11818) | `--cache-ram 384`, `mem_limit: 4608m` |
| Quota `cpus: "3.6"` con 3–4 thread ggml: il throttling CFS blocca le barriere in busy-wait | `deploy_n100/docker-compose.yml` | quota rimossa, `cpu_shares` + `--prio 2` |

La KV cache `q8_0` non è la causa principale (è quasi lossless), ma con 12 GB liberati da Ollama si torna a F16
per togliere una variabile. Lo speculative decoding ngram non era più attivo.

### 2. Prompt leakage e risposte fisse (`"È un pazienato da fare…"`)

Sono quattro cause che si sommano:

1. **Template di chat diverso dal training.** Il template nel GGUF in produzione (verificato con `gguf-py`) produce
   `{system}<用户>{utente}<AI>{risposta}<用户>…`: il system sta *fuori* dal turno e tra i turni non c'è `</s>`.
   Il training usava `<用户>{system}\n\n{utente}<AI>{risposta}</s>`. → `deploy_n100/minicpm-chat.jinja`, verificato
   carattere per carattere uguale a `render_minicpm()` usato in training.
2. **History non filtrata.** `/api/ai-chat/c3po-query` inoltrava tutta la cronologia, comprese le vecchie risposte con
   `![…](https://image.pollinations.ai/…)`; un 2B le ricopia. → `sanitizeHistory()` toglie immagini, link ai media,
   card e blocchi "Fonti", e azzera la history per RAG, PDF e immagini.
3. **System prompt "tuttofare"** con regole SOC, metriche fisse ("RAM 3GB/11GB"), esempi `![...](https://...)` e
   regole sui file nello stesso testo. → un prompt statico e breve per intent (`lib/prompts.json`), senza esempi.
4. **Dataset sovra-ripetuto.** `train_multitask.json` ha 719 risposte uniche su 1425 esempi; 6 risposte sono
   ripetute ~75 volte ciascuna (circa un terzo dei dati). Nel test locale il modello attuale risponde
   "nginx/unhealthy" quando la telemetria dice `mail-app`, e aggiunge "Cgroups 2.0" a fonti che non lo citano.

### 3. Latenza e italiano

- La telemetria nel system prompt cambiava a ogni richiesta e invalidava la cache del prefisso. Ora il system è
  statico e i dati vanno nel turno user (`[TELEMETRIA]`, `[FONTI]`, `[DOMANDA]`), in formato chiave=valore (~60 token).
- Il fast-path esistente rispondeva con **valori inventati** quando la telemetria mancava ("23 container", "39°C"), e
  il bollettino di sicurezza era interamente hardcoded ("0 tentativi SSH", "15 IoC"). Ora usa solo dati reali, oppure
  "n/d".
- La qualità dell'italiano dipende dai pesi: va migliorata col retraining (sezione Modulo 5).

### Altri problemi trovati

- **SSRF:** il vecchio `web-research-service` scaricava qualsiasi URL scritto in chat, compresi i servizi interni
  (`http://docker-proxy:2375/...`). Ora filtra schemi, porte, nomi senza dominio, IP privati e DNS verso reti interne,
  anche sui redirect.
- **Visione:** senza Ollama, `describeImageWithVision` restituiva "Immagine analizzata con successo" come descrizione:
  il modello inventava il contenuto. Ora lo dice esplicitamente (configurabile con `VISION_HOST`).
- **PDF:** `switchToPage()` senza `bufferPages: true` e il piè di pagina scritto sotto il margine (pagine bianche);
  il Markdown veniva stampato grezzo. → `lib/markdown-pdf.js`.
- **Merge:** `post_train_multitask.py` forzava `tie_word_embeddings = True`. È corretto per `MiniCPM-2B-dpo-fp16`, ma
  **sbagliato per `MiniCPM-2B-dpo-bf16-llama-format`** (lm_head separato): su quel modello il merge produrrebbe logit errati.
- **imatrix** calibrata su testo ChatML (`<|im_start|>`), un formato che il modello non vede mai.
- **Credenziali:** la password SSH era in chiaro in 4 script, già pubblicati su GitHub. Rimossa dal codice, **ma va
  cambiata** (resta nella cronologia git). Meglio passare alle chiavi: `ssh-copy-id gabibbo@192.168.1.180`.

## Architettura

```
chat-app ──JSON──► /api/ai-chat/c3po-query ─┐
web UI   ──SSE───► /api/ai-chat/stream ─────┤
                                            ▼
                        lib/c3po-orchestrator.js
                        classifyIntent()  (lib/intent-router.js, regex, <1 ms)
     ┌─────────────┬──────────────┬───────────────┬──────────────┬───────────────┐
 INTENT_CHAT    INTENT_SOC     INTENT_WEB_SEARCH  INTENT_PDF     INTENT_IMAGE
 prompt chat    telemetria     DDG→scrape→[1..3]  Markdown→PDF   IT→EN (LLM, t=0)
 history pulita k=v nel turno  link dal codice    markdown-pdf   Pollinations
     └─────────────┴──────────────┴───────┬───────┴──────────────┘
                                          ▼
                 lib/ai-engine-client.js (profili di campionamento, stop, sanitizer)
                                          ▼
              minicpm2b-n100: llama-server + minicpm-chat.jinja (2 slot × 2048 tok)
```

Priorità degli intent: PDF/DOCX > IMMAGINE > RICERCA WEB > SOC > CHAT. Con un allegato si resta in CHAT.

### Profili di campionamento

| Profilo | temp | top_p | top_k | min_p | repeat_penalty | max_tokens |
|---|---|---|---|---|---|---|
| chat | 0.5 | 0.8 | 40 | 0.05 | 1.05 | 320 |
| live (voce) | 0.4 | 0.8 | 40 | 0.05 | 1.05 | 90 |
| soc | 0.2 | 0.8 | 20 | 0.05 | 1.05 | 256 |
| rag | 0.2 | 0.8 | 20 | 0.05 | 1.05 | 320 |
| doc | 0.35 | 0.85 | 40 | 0.05 | 1.07 | 750 |
| translate | 0.0 | 1.0 | 1 | 0 | 1.0 | 60 |

`presence_penalty` e `frequency_penalty` sono sempre 0. Stop: `<用户>`, `<AI>`, `</s>`.

## File

| File | Contenuto |
|---|---|
| `ai-soc-app/lib/prompts.json` | System prompt statici, condivisi con `finetune/prepare_multitask_dataset.py` |
| `ai-soc-app/lib/ai-engine-client.js` | Client llama-server: profili, `cache_prompt`, sanitizer, stop dei loop in streaming |
| `ai-soc-app/lib/intent-router.js` | Router deterministico + estrazione di query e argomento |
| `ai-soc-app/lib/prompt-builder.js` | History ripulita, telemetria compatta, turno user strutturato |
| `ai-soc-app/lib/image-intent.js` | Soggetto, stile, traduzione validata con fallback a dizionario |
| `ai-soc-app/lib/web-research-service.js` | DDG HTML → DDG Lite → Wikipedia, `cheerio` con fallback regex, SSRF guard, cache 10 min |
| `ai-soc-app/lib/markdown-pdf.js` | Parser Markdown → PDF A4 (pdfkit): titoli, grassetti inline, elenchi, tabelle multipagina, codice |
| `ai-soc-app/lib/c3po-orchestrator.js` | Orchestrazione degli intent, `run()` JSON e `stream()` SSE |
| `ai-soc-app/lib/ai-chat-service.js` | Versione del server con generazione delegata all'orchestratore |
| `ai-soc-app/lib/ollama-service.js` | SOC Sentinel stream con system statico |
| `ai-soc-app/lib/fast-path.js` | Risposte istantanee solo con dati reali |
| `deploy/deploy_home_server.py` | Upload, patch di `server.js` / `media-studio.js` / `package.json`, riavvio |

## Operatività

### Test locali (Windows)

```bash
cd home-server/ai-soc-app && npm install && npm test
```

### 1. Runtime llama-server (sul server)

```bash
cd /home/gabibbo/minicpm && git pull
```

```bash
cd /home/gabibbo/minicpm/deploy_n100 && docker compose up -d --force-recreate llama-server
```

Verifica che il template sia quello nuovo (deve iniziare con `{#-` e contenere `ns.sys`):

```bash
curl -s http://127.0.0.1:8080/props | python3 -c "import sys,json; print(json.load(sys.stdin)['chat_template'][:300])"
```

### 2. ai-soc-app (da Windows)

Prova a secco su copie locali dei file del server (nomi `ai-soc-app__server.js`, ecc.):

```bash
python home-server/deploy/deploy_home_server.py --offline <cartella>
```

Deploy: con `--rebuild` viene installato anche `cheerio`. Senza `--rebuild` lo scraping usa il fallback regex.

```bash
python home-server/deploy/deploy_home_server.py --rebuild
```

Prova end-to-end dei 5 intent:

```bash
python test_all_scenarios.py
```

Il deploy crea un backup `.bak-<timestamp>` di ogni file toccato. **HOME-SERVER è un repository git** (`avvia_server.sh`
esegue `git pull`): dopo la verifica, committa le modifiche nel repository HOME-SERVER, altrimenti il prossimo pull
può rifiutarsi di partire o sovrascriverle.

### 3. Retraining (RTX 2080)

```bash
python finetune/prepare_multitask_dataset.py --total 4000
```

```powershell
.\finetune\run_qlora_win.ps1
```

```bash
python finetune/post_train_multitask.py --ppl
```

`post_train_multitask.py` esegue merge, GGUF F16, imatrix sul formato reale, `Q4_K_M` (oppure `--quant IQ4_XS`) e
uno smoke test che blocca il modello se l'output contiene byte rotti, cicli, marcatori di turno o immagini inventate.
Con `--deploy` carica il file e imposta `MINICPM_MODEL` in `deploy_n100/.env`.

`Q4_K_M` contro `IQ4_XS` sul N100: IQ4_XS è più piccolo del 6% circa, quindi un po' più veloce in generazione (che è
limitata dalla banda RAM), con una perdita di qualità di solito trascurabile se c'è l'imatrix. Va misurato sul N100:

```bash
llama-bench -m /models/minicpm-2b-q4_k_m.gguf -m /models/minicpm-2b-iq4_xs.gguf -t 3 -p 256 -n 64
```

## Limiti noti

- Slot da 2048 token: allegati e file letti vengono troncati a ~2400 caratteri.
- Il filtro SSRF risolve il DNS prima della richiesta; un DNS che cambia risposta tra controllo e fetch (rebinding)
  non è coperto. Per una casa va bene; con utenti non fidati serve un proxy in uscita.
- DuckDuckGo HTML può cambiare markup o limitare le richieste: in quel caso la ricerca ricade su Wikipedia.
- MiniCPM-2B resta un modello piccolo con un italiano limitato anche dopo il retraining.
