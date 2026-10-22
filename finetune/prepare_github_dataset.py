# -*- coding: utf-8 -*-
"""
Script per la preparazione del dataset basato su progetti GitHub reali (Magicoder, Evol-Instruct)
combinato con diagnostica server Linux/Docker/N100, Cybersecurity/SOC (C-3PO) e ragionamento in lingua italiana.

Regola ferrea di efficienza:
- Risposte ultra-tecniche, chirurgiche e orientate al codice/soluzione.
- Massimo 40-60 token per risposta.
- Zero frasi di cortesia ("Certamente", "Ecco a te", "In qualità di...") o preamboli prolissi.
"""

import os
import sys
import json
import re
import random
import argparse
from typing import List, Dict, Any

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
    "Sei C-3PO, assistente tecnico e droide protocollare di HOME-SERVER (192.168.1.180, CPU Intel N100 4-core, 12GB RAM).\n"
    "Stile High-Density: rispondi in modo chirurgico, ultra-conciso e orientato al codice o alla risoluzione immediata (massimo 40-60 token). "
    "Zero convenevoli o preamboli."
)

PREAMBLE_PATTERNS = [
    r"^(?:Certo,?|Certamente,?!|Ecco,?|Ecco a te,?|Di seguito,?|Sicuramente,?|"
    r"In qualità di assistente,?|Come assistente IA,?|Sì,? posso aiutarti|Volentieri,?|"
    r"Ecco un'analisi dettagliata:?|Certamente gabibbo!?|Salve!?|Ciao!?|"
    r"Here is the code:?|Sure,? here is the solution:?|Certainly!?|Below is the implementation:?|"
    r"To achieve this,?|You can solve this by:?)\s*",
    r"^(?:Here's how you can do it:?|Here is an example:?|Here's an example:?)\s*"
]

COMBINED_PREAMBLE_REGEX = re.compile("|".join(PREAMBLE_PATTERNS), re.IGNORECASE)


def clean_response(text: str, max_chars: int = 280) -> str:
    """Rimuove convenevoli, preamboli ed estrae risposte ultra-concise (< 60 token)."""
    cleaned = COMBINED_PREAMBLE_REGEX.sub("", text.strip()).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Se c'è un blocco di codice, manteniamo il codice essenziale
    code_match = re.search(r"```(?:\w+)?\n([\s\S]*?)```", cleaned)
    if code_match:
        code_body = code_match.group(1).strip()
        lines = [line for line in code_body.split("\n") if line.strip() and not line.strip().startswith("//") and not line.strip().startswith("#")]
        if len(lines) <= 6:
            return f"```\n{code_body}\n```"
    
    # Taglio per frasi
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    shortened = ""
    for s in sentences:
        if len(shortened) + len(s) + 1 <= max_chars:
            shortened = (shortened + " " + s).strip()
        else:
            break

    if not shortened:
        shortened = cleaned[:max_chars].strip()

    return shortened


def generate_server_soc_samples() -> List[Dict[str, Any]]:
    """Campioni di alta densità per Linux, Docker, N100 e Cybersecurity/SOC."""
    samples = []

    soc_and_server = [
        # CPU & Termiche Intel N100
        ("La temperatura della CPU è a 48°C. Stato del processore?",
         "Temperatura a 48°C ottimale per Intel N100 (Tjmax 105°C). Nessun thermal throttling; i 4 core Gracemont operano a freddo con consumo sotto i 6W."),
        
        ("La CPU dell'N100 tocca 58°C sotto carico di inferenza.",
         "Temperatura normale sotto carico: ampiamente entro i margini di sicurezza. Nessun intervento richiesto."),
         
        ("[METRICHE: cpu=42C ram=4.2/11G up=23/23]\nAnalisi telemetria host.",
         "Stato nominale: CPU 42°C eccellente, RAM occupata al 38% (6.8 GB liberi) e 23 container attivi su 23."),

        # Docker & Container Linux
        ("Come monitorare in tempo reale l'uso di risorse dei container?",
         "Usa `docker stats --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}'` per telemetria compatta a terminale."),

        ("Un container si riavvia con Exit Code 137. Causa e fix?",
         "Exit Code 137 è SIGKILL per OOM-Killer. Aumenta `mem_limit` in docker-compose.yml o analizza leak con `dmesg -T | grep -i oom`."),

        ("Come ripulire immagini e volumi orfani senza fermare lo stack?",
         "Esegui `docker system prune -a --volumes -f` per recuperare spazio su disco eliminando container fermi e layer inutilizzati."),

        ("Come forzare il riavvio rapido di un singolo servizio compose?",
         "Esegui `docker compose restart <servizio>` dalla cartella di deployment; le connessioni di rete degli altri container rimangono intatte."),

        # Nginx & Reverse Proxy
        ("Perché impostare 'proxy_buffering off;' con llama-server e SSE?",
         "Impedisce a Nginx di bufferizzare i token in streaming, trasmettendo i chunk SSE all'interfaccia web con latenza zero."),

        ("Come verificare la validità del certificato TLS locale da CLI?",
         "Esegui `echo | openssl s_client -connect 127.0.0.1:443 -servername localhost 2>/dev/null | openssl x509 -noout -dates`."),

        # SQLite & Database
        ("Errore SQLite: 'database is locked'. Risoluzione immediata?",
         "Abilita la modalità WAL con `sqlite3 file.db 'PRAGMA journal_mode=WAL;'` e imposta `PRAGMA busy_timeout = 5000;` per sbloccare la concorrenza in lettura."),

        # SOC Incident Response & Security
        ("Rilevati 50 tentativi SSH falliti da IP 194.26.29.112.",
         "Brute force isolato. Blocca l'IP con `fail2ban-client set sshd banip 194.26.29.112` o `iptables -I INPUT -s 194.26.29.112 -j DROP`."),

        ("Rilevato scan di vulnerabilità su porte esposte di Nginx.",
         "Isola gli IP sorgente da `access.log` e applica rate limiting su `nginx.conf`: `limit_req zone=one burst=10 nodelay;`."),

        ("Come visualizzare le connessioni TCP attive su porte server?",
         "Esegui `ss -tulnp` per visualizzare socket in ascolto e relativi PID dei processi."),

        # Identità C-3PO
        ("Chi sei?",
         "Sono C-3PO, droide protocollare e assistente operativo per HOME-SERVER su Intel N100. Fornisco diagnostica immediata e risoluzione guasti."),

        ("Qual è l'architettura di questo server?",
         "Intel Alder Lake-N100 (4 core Gracemont, 12 GB RAM DDR4, SSD NVMe), Ubuntu Server 24.04 e stack Docker per microservizi.")
    ]

    for q, a in soc_and_server:
        # Duplica i casi critici per ancorare i pesi
        weight = 8 if "Exit Code 137" in q or "48°C" in q or "C-3PO" in q else 5
        for _ in range(weight):
            samples.append({
                "messages": [
                    {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    return samples


def stream_github_code_samples(max_samples: int = 800) -> List[Dict[str, Any]]:
    """Estrae problemi di coding e soluzioni concise da Magicoder ed Evol-Instruct (GitHub OSS)."""
    samples = []
    if load_dataset is None:
        return samples

    print(f"--> Streaming campioni di codice GitHub OSS da Hugging Face...")

    # 1. Magicoder OSS Instruct
    try:
        ds_magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
        count = 0
        for item in ds_magic:
            prob = (item.get("problem") or "").strip()
            sol = (item.get("solution") or "").strip()
            if not prob or not sol:
                continue
            # Seleziona problemi compatti e soluzioni concise
            if len(prob) < 300 and len(sol) < 400:
                clean_sol = clean_response(sol, max_chars=250)
                if len(clean_sol) > 15:
                    samples.append({
                        "messages": [
                            {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                            {"role": "user", "content": prob},
                            {"role": "assistant", "content": clean_sol}
                        ]
                    })
                    count += 1
                    if count >= max_samples // 2:
                        break
        print(f"    Magicoder campioni estratti: {count}")
    except Exception as e:
        print(f"    Avviso caricamento Magicoder: {e}")

    # 2. Evol-Instruct-Code
    try:
        ds_evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        count = 0
        for item in ds_evol:
            inst = (item.get("instruction") or "").strip()
            out = (item.get("output") or "").strip()
            if not inst or not out:
                continue
            if len(inst) < 250 and len(out) < 400:
                clean_out = clean_response(out, max_chars=250)
                if len(clean_out) > 15:
                    samples.append({
                        "messages": [
                            {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                            {"role": "user", "content": inst},
                            {"role": "assistant", "content": clean_out}
                        ]
                    })
                    count += 1
                    if count >= max_samples // 2:
                        break
        print(f"    Evol-Instruct campioni estratti: {count}")
    except Exception as e:
        print(f"    Avviso caricamento Evol-Instruct: {e}")

    return samples


def stream_italian_samples(max_samples: int = 700) -> List[Dict[str, Any]]:
    """Estrae e distilla campioni in lingua italiana da saillab/alpaca-italian-cleaned."""
    samples = []
    if load_dataset is None:
        return samples

    print(f"--> Caricamento campioni italiani da saillab/alpaca-italian-cleaned...")
    try:
        ds_ita = load_dataset("saillab/alpaca-italian-cleaned", split="train", streaming=True)
        count = 0
        for item in ds_ita:
            inst = (item.get("instruction") or "").strip()
            inp = (item.get("input") or "").strip()
            out = (item.get("output") or "").strip()

            if not inst or not out:
                continue

            user_q = f"{inst}\n{inp}".strip() if inp else inst
            clean_out = clean_response(out, max_chars=220)

            # Evita risposte vuote o troppo brevi
            if len(clean_out) >= 20 and len(user_q) < 300:
                samples.append({
                    "messages": [
                        {"role": "system", "content": HIGH_DENSITY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": clean_out}
                    ]
                })
                count += 1
                if count >= max_samples:
                    break
        print(f"    Campioni italiani estratti: {count}")
    except Exception as e:
        print(f"    Avviso caricamento italiano: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepara dataset GitHub + Server + Italiano per MiniCPM-2B")
    parser.add_argument("--output_dir", type=str, default="finetune/data")
    parser.add_argument("--target_total", type=int, default=1800)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 65)
    print("MiniCPM-2B: Generazione Dataset GitHub OSS + C-3PO High-Density")
    print("=" * 65)

    all_samples = []

    # 1. Server, N100, Docker & SOC Sentinel
    server_samples = generate_server_soc_samples()
    all_samples.extend(server_samples)
    print(f"--> Campioni Server / Docker / SOC: {len(server_samples)}")

    # 2. GitHub Code OSS Instruct
    code_samples = stream_github_code_samples(max_samples=850)
    all_samples.extend(code_samples)

    # 3. Italian Distilled Reasonings
    ita_samples = stream_italian_samples(max_samples=750)
    all_samples.extend(ita_samples)

    print(f"\n--> Totale campioni raccolti: {len(all_samples)}")

    random.seed(args.seed)
    random.shuffle(all_samples)

    if len(all_samples) > args.target_total:
        all_samples = all_samples[:args.target_total]

    # Split 95% Train / 5% Eval
    num_eval = max(1, int(len(all_samples) * args.eval_ratio))
    eval_data = all_samples[:num_eval]
    train_data = all_samples[num_eval:]

    os.makedirs(args.output_dir, exist_ok=True)

    # Salva sia train_custom/eval_custom che train_github/eval_github per massima compatibilità
    for prefix in ["train_custom", "train_github"]:
        p = os.path.join(args.output_dir, f"{prefix}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

    for prefix in ["eval_custom", "eval_github"]:
        p = os.path.join(args.output_dir, f"{prefix}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("=" * 65)
    print("SUCCESS: Dataset generati con successo!")
    print(f"--> Training set:   {len(train_data)} esempi")
    print(f"--> Validation set: {len(eval_data)} esempi")
    print(f"--> File salvati in: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
