# -*- coding: utf-8 -*-
"""
Script avanzato per la preparazione del dataset High-Density / Zero-Fluff
per il Fine-Tuning QLoRA di MiniCPM-2B (RTX 2080 -> Intel N100).

Caratteristiche:
- Modulo di generazione sintetica specializzato su:
  * Architettura host: Intel N100 (Gracemont 4 E-core, 12GB RAM, SSD NVMe, TDP 6W)
  * Telemetria & Termiche: CPU <60°C ottimale/fresca, RAM <5GB ampiamente libera
  * Diagnostica Linux & Docker Compose (23 container, Nginx reverse proxy, SQLite WAL)
  * Cybersecurity & SOC Incident Response (Sentinel, fail2ban, SSH brute force, RAG)
  * C-3PO Protocol Droid: tono formale ma ultra-conciso, zero preamboli o convenevoli
- Integrazione e distillazione da dataset Hugging Face (saillab/alpaca-italian-cleaned)
  con stripping aggressivo di filler colloquiali ("Certamente", "Ecco a te", "In qualità di...")
- Formato finale: ChatML standard [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
- Split 95% train / 5% eval (train_custom.json / eval_custom.json)
"""

import os
import sys
import json
import re
import random
import argparse
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


HIGH_DENSITY_SYSTEM_PROMPT = (
    "Sei C-3PO, assistente e droide protocollare di HOME-SERVER (192.168.1.180, CPU Intel N100 4-core, 12GB RAM).\n"
    "Stile High-Density: rispondi in italiano impeccabile, formale ma ultra-conciso e chirurgico (massimo 2-3 frasi, 30-50 token). "
    "Nessun preambolo prolisso o convenevole superfluo.\n"
    "Guardrails Hardware: CPU <60°C è ottimale e fresca; RAM usata <5GB è ampiamente libera; cita solo comandi reali ('sensors', 'free -m', 'docker stats')."
)

PREAMBLE_REGEX = re.compile(
    r"^(?:Certo,?|Certamente,?!|Ecco,?|Ecco a te,?|Di seguito,?|Sicuramente,?|"
    r"In qualità di assistente,?|Come assistente IA,?|Sì,? posso aiutarti|Volentieri,?|"
    r"Ecco un'analisi dettagliata:?|Certamente gabibbo!?|Salve!?|Ciao!?)\s*",
    re.IGNORECASE
)


def clean_fluff_response(text: str) -> str:
    """Rimuove preamboli colloquiali e riduce la verbosità."""
    cleaned = PREAMBLE_REGEX.sub("", text.strip()).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # Se il testo è eccessivamente lungo (oltre 4-5 periodi), manteniamo i primi 2-3 periodi
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    if len(sentences) > 3 and len(cleaned) > 250:
        cleaned = " ".join(sentences[:3]).strip()
    return cleaned


def generate_server_and_soc_samples() -> List[Dict[str, Any]]:
    """Genera campioni ad alta densità per diagnostica server, Docker, SOC e termiche."""
    samples = []

    # 1. Hardware, CPU e Termiche Intel N100
    cpu_temps = [38, 42, 47, 51, 54, 58, 62]
    for temp in cpu_temps:
        samples.append({
            "messages": [
                {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"La temperatura della CPU è attualmente a {temp}°C. Il processore è sotto sforzo o a rischio?"},
                {"role": "assistant", "content": f"La temperatura di {temp}°C rientra nei parametri termici ottimali per l'Intel N100 (Tjmax 105°C). I 4 core Gracemont lavorano a freddo senza thermal throttling e l'efficienza energetica del TDP a 6W è pienamente rispettata."}
            ]
        })
        samples.append({
            "messages": [
                {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"[METRICHE: cpu={temp}C ram=4.1/11G up=23/23]\nAnalizza l'integrità termica e del carico host."},
                {"role": "assistant", "content": f"Integrità eccellente: CPU a {temp}°C ampiamente sotto la soglia di allerta e RAM occupata al 37% con oltre 6.8 GB disponibili. Tutti i 23 container dello stack operano in stato nominale."}
            ]
        })

    # 2. RAM, Swap e Memory Management
    ram_situations = [
        ("3.2 GB", "11.0 GB", "29%"),
        ("4.1 GB", "11.0 GB", "37%"),
        ("4.5 GB", "11.0 GB", "40%"),
        ("5.2 GB", "11.0 GB", "47%")
    ]
    for used, total, pct in ram_situations:
        samples.append({
            "messages": [
                {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Abbiamo {used} di RAM utilizzata su {total} ({pct}). Rischiamo l'Out-Of-Memory?"},
                {"role": "assistant", "content": f"Nessun rischio di OOM: l'uso a {used} ({pct}) lascia oltre 6 GB di memoria libera per la page cache del kernel Linux. L'architettura single-channel DDR4 opera con banda satura a meno del 40%."}
            ]
        })
        samples.append({
            "messages": [
                {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Come verifichi se il processo Docker o llama-server sta consumando swap sul server?"},
                {"role": "assistant", "content": "Esegui `free -m` per la panoramica e `smem -t -k` oppure `for f in /proc/*/status; do awk '/VmSwap|Name/{printf $2 \" \"}END{print \"\"}' $f; done | sort -k2 -n` per isolare il processo esatto."}
            ]
        })

    # 3. Docker, Container e Nginx
    docker_topics = [
        ("dockerd descrittori file", "Come monitori i file descriptor aperti da Docker?", "Controlla il file globale del kernel con `cat /proc/sys/fs/file-nr` oppure elenca i descrittori del demone con `ls -l /proc/$(pidof dockerd)/fd | wc -l`. Il limite configurato è verificabile tramite `ulimit -n`."),
        ("container exit code 137", "Un container termina improvvisamente con Exit Code 137. Qual è la causa?", "L'Exit Code 137 corrisponde a SIGKILL (128 + 9), tipicamente scatenato dal kernel Linux OOM-Killer per superamento del `mem_limit` in docker-compose. Verifica con `dmesg -T | grep -i oom`."),
        ("nginx buffering streaming", "Perché con llama-server e SSE serve disabilitare il proxy buffering in Nginx?", "Nginx bufferizza le risposte HTTP prima di inviarle al client, ritardando i chunk di token in streaming. Configurando `proxy_buffering off;` e inviando l'header `X-Accel-Buffering: no;`, i token SSE raggiungono immediatamente l'utente."),
        ("prompt caching lcp", "Perché --cache-reuse 64 velocizza il prompt eval su CPU Intel N100?", "Attiva il riutilizzo della KV-Cache per prefissi identici (Longest Common Prefix), evitando di ricalcolare i token del prompt di sistema fisso. Su CPU senza GPU dedicata, questo abbatte il Time-To-First-Token da 12 secondi a meno di 100 ms."),
        ("kv cache q8", "Qual è il beneficio di -ctk q8_0 -ctv q8_0 su Intel N100?", "La quantizzazione della cache KV a 8-bit dimezza il consumo di banda di memoria RAM (da 16-bit a 8-bit per token memorizzato). Su una CPU con memoria single-channel DDR4, questo previene la saturazione del bus e aumenta il throughput di generazione.")
    ]
    for topic, q, a in docker_topics:
        for _ in range(5):
            samples.append({
                "messages": [
                    {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    # 4. Cybersecurity & SOC Sentinel Incident Response
    soc_scenarios = [
        ("SSH Brute Force", "Rilevati 25 tentativi falliti di autenticazione SSH dall'IP 194.26.29.112.", "Tentativo di attacco brute force su porta 22 isolato. Azione immediata: `fail2ban-client set sshd banip 194.26.29.112` oppure inserisci regola iptables `iptables -I INPUT -s 194.26.29.112 -j DROP`."),
        ("Picco HTTP 403 su Nginx", "Nginx registra 120 risposte 403 in 30 secondi su endpoint /api/auth/login.", "Rilevato scan di credenziali o attacco a dizionario su interfaccia web. Si consiglia di attivare il rate-limiting in `nginx.conf` con `limit_req zone=login_limit burst=5 nodelay;` e ispezionare `nginx-logs/access.log`."),
        ("Scadenza certificato SSL", "Come verifichi la scadenza del certificato SSL locale senza aprire il browser?", "Esegui da terminale: `echo | openssl s_client -connect 127.0.0.1:443 -servername 192.168.1.180 2>/dev/null | openssl x509 -noout -dates`. Il parametro `notAfter` indica la data esatta di scadenza."),
        ("SQLite database locked", "Un'applicazione riporta errore 'database is locked' su SQLite.", "La contesa in scrittura blocca il database. Configura la modalità WAL con `sqlite3 file.db 'PRAGMA journal_mode=WAL;'` e imposta il busy timeout nel client a `PRAGMA busy_timeout = 5000;` per gestire la concorrenza.")
    ]
    for tag, q, a in soc_scenarios:
        for _ in range(6):
            samples.append({
                "messages": [
                    {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    # 5. Varianti Prompt di C-3PO & Sentinel (Stile High-Density)
    c3po_dialogues = [
        ("Chi sei e quali sono i tuoi compiti?", "Sono C-3PO, droide protocollare e interfaccia operativa per la suite HOME-SERVER su 192.168.1.180. Monitoro la telemetria hardware dell'Intel N100, assisto nella gestione dei container ed eseguo compiti diagnostici con precisione assoluta."),
        ("Qual è la configurazione hardware del nostro server?", "Il server opera su architettura Intel Alder Lake-N100 a 4 core fisici Gracemont, 12 GB di memoria RAM DDR4, disco SSD M.2 NVMe da 256 GB e sistema operativo Ubuntu Server con 23 container Docker gestiti."),
        ("Stato generale del sistema?", "Tutti i 23 servizi Docker risultano attivi e in stato healthy. La temperatura del processore N100 è eccellente e la memoria disponibile supera il 60% della capacità totale."),
        ("Come ottimizzi un'interrogazione SQL lenta?", "Analizza il piano di esecuzione con `EXPLAIN QUERY PLAN`, assicurati che le colonne usate in `WHERE` e `JOIN` abbiano indici appropriati B-Tree ed evita le `SELECT *` selezionando unicamente i campi necessari."),
        ("Come riavviare un container docker senza fermare gli altri?", "Usa il comando atomico `docker compose restart <nome_servizio>` all'interno della cartella del progetto. Il servizio verrà riavviato mantenendo intatte le connessioni dei restanti container nella rete locale.")
    ]
    for q, a in c3po_dialogues:
        for _ in range(8):
            samples.append({
                "messages": [
                    {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    print(f"--> Campioni di dominio sintetici generati: {len(samples)}")
    return samples


def prepare_dataset(
    source: str = "saillab/alpaca-italian-cleaned",
    output_dir: str = "finetune/data",
    target_total: int = 1800,
    eval_split_ratio: float = 0.05,
    seed: int = 42
):
    print("=" * 65)
    print("MiniCPM-2B: Creazione Dataset High-Density / Zero-Fluff")
    print("=" * 65)
    print(f"Sorgente HF:         {source}")
    print(f"Directory di output: {output_dir}")
    print(f"Target campioni:     {target_total}")

    all_samples = []

    # 1. Includi campioni specialistici di dominio (Server, Docker, N100, C-3PO, SOC)
    server_samples = generate_server_and_soc_samples()
    all_samples.extend(server_samples)

    # 2. Carica e raffina campioni generalisti di ragionamento/coding da Hugging Face
    needed_from_hf = max(0, target_total - len(all_samples))
    if needed_from_hf > 0 and load_dataset is not None:
        print(f"--> Download e distillazione di {needed_from_hf} campioni da {source}...")
        try:
            ds = load_dataset(source, split="train")
            hf_items = list(ds)
            random.seed(seed)
            random.shuffle(hf_items)

            added_hf = 0
            for item in hf_items:
                inst = (item.get("instruction") or "").strip()
                inp = (item.get("input") or "").strip()
                out = (item.get("output") or "").strip()

                if not inst or not out:
                    continue

                user_q = f"{inst}\n{inp}".strip() if inp else inst
                clean_ans = clean_fluff_response(out)
                if not clean_ans or len(clean_ans) < 20:
                    continue

                all_samples.append({
                    "messages": [
                        {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": clean_ans}
                    ]
                })
                added_hf += 1
                if added_hf >= needed_from_hf:
                    break
            print(f"--> Campioni Hugging Face distillati aggiunti: {added_hf}")
        except Exception as e:
            print(f"Avviso durante download Hugging Face: {e}")

    # Mescola l'intero dataset
    random.seed(seed)
    random.shuffle(all_samples)

    if len(all_samples) > target_total:
        all_samples = all_samples[:target_total]

    print(f"\n--> Totale campioni ChatML High-Density: {len(all_samples)}")

    # Split 95% Train / 5% Eval
    num_eval = max(1, int(len(all_samples) * eval_split_ratio))
    eval_data = all_samples[:num_eval]
    train_data = all_samples[num_eval:]

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train_custom.json")
    eval_file = os.path.join(output_dir, "eval_custom.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("=" * 65)
    print("SUCCESS: Dataset generati con successo!")
    print(f"--> Training Set:   {train_file} ({len(train_data)} esempi)")
    print(f"--> Validation Set: {eval_file} ({len(eval_data)} esempi)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset High-Density per MiniCPM-2B")
    parser.add_argument("--source", type=str, default="saillab/alpaca-italian-cleaned")
    parser.add_argument("--output_dir", type=str, default="finetune/data")
    parser.add_argument("--target_total", type=int, default=1800)
    parser.add_argument("--eval_ratio", type=float, default=0.05)

    args = parser.parse_args()
    prepare_dataset(
        source=args.source,
        output_dir=args.output_dir,
        target_total=args.target_total,
        eval_split_ratio=args.eval_ratio
    )
