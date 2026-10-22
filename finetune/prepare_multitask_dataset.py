# -*- coding: utf-8 -*-
"""
Multi-Task High-Density Dataset Builder for MiniCPM-2B (RTX 2080 -> Intel N100).
Task Distribution:
- 40% Italian Reasoning & Instructions (saillab/alpaca-italian-cleaned)
- 30% Web RAG & Autonomous Tool Use (Structured search context, factual extraction, citation)
- 15% Technical Markdown / PDF Reports (#, ##, * bullets, strict clean formatting)
- 15% Coding, Linux Sysadmin & Cybersecurity SOC (Magicoder + Sentinel incident response)
"""

import os
import sys
import json
import re
import random
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from datasets import load_dataset

CHAT_SYSTEM_PROMPT = (
    "Sei C-3PO, un assistente AI intelligente, chiaro, preciso e cordiale per HOME-SERVER.\n"
    "Rispondi in italiano corretto, naturale e pertinente alla richiesta dell'utente, senza inventare dati."
)

PREAMBLE_REGEX = re.compile(
    r"^(?:Certo,?|Certamente,?!|Ecco,?|Ecco a te,?|Di seguito,?|Sicuramente,?|"
    r"In qualità di assistente,?|Come assistente IA,?|Sì,? posso aiutarti|Volentieri,?|"
    r"Ecco un'analisi dettagliata:?|Certamente!?|Salve!?|Ciao!?)\s*",
    re.IGNORECASE
)

def clean_fluff(text: str) -> str:
    cleaned = PREAMBLE_REGEX.sub("", text.strip()).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned

# =========================================================================
# PILLAR 1: Italian Reasoning & Instructions (40%)
# =========================================================================
def build_italian_reasoning_samples(target_count: int = 600, seed: int = 42) -> List[Dict[str, Any]]:
    print(f"--> [Pillar 1: 40%] Generazione {target_count} campioni di Italiano & Ragionamento...")
    ds = load_dataset("saillab/alpaca-italian-cleaned", split="train")
    items = list(ds)
    random.seed(seed)
    random.shuffle(items)

    samples = []
    for it in items:
        instr = (it.get("instruction") or "").strip()
        inp = (it.get("input") or "").strip()
        out = (it.get("output") or "").strip()
        if not instr or not out:
            continue
        if len(out) < 20 or len(out) > 1200:
            continue

        user_content = f"{instr}\n{inp}".strip() if inp else instr
        ans_content = clean_fluff(out)

        samples.append({
            "messages": [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ans_content}
            ]
        })
        if len(samples) >= target_count:
            break
    print(f"    Completato Pillar 1: {len(samples)} campioni")
    return samples

# =========================================================================
# PILLAR 2: Web RAG & Autonomous Tool Use (30%)
# =========================================================================
def build_web_rag_samples(target_count: int = 450) -> List[Dict[str, Any]]:
    print(f"--> [Pillar 2: 30%] Generazione {target_count} campioni di Web RAG & Citazioni...")
    topics = [
        ("Quali sono le novità del kernel Linux 6.12?", 
         "Linux Kernel Organization (https://kernel.org)", 
         "Il kernel Linux 6.12 introduce il supporto definitivo per lo scheduler sched_ext (Bcachefs e eBPF extensible scheduler), miglioramenti per l'architettura ARM64 e driver grafici Intel Xe e AMD RDNA4.",
         "Le principali novità di Linux 6.12 includono l'integrazione ufficiale dello scheduler estensibile sched_ext basato su eBPF, ottimizzazioni energetiche per processori moderni e driver aggiornati per GPU Intel Xe e AMD.\n\n**Fonti consultate:**\n- [Linux Kernel Organization](https://kernel.org)"),
        
        ("Cosa è successo nell'ultima release di Docker Engine 27?",
         "Docker Docs (https://docs.docker.com/engine/release-notes/)",
         "Docker Engine 27.x introduce Buildx integrato di default, supporto per Containerd image store nativo e accelerazione per build multi-architettura.",
         "Docker Engine 27 abilita lo storage nativo delle immagini tramite containerd e integra Docker Buildx con supporto multi-piattaforma ottimizzato.\n\n**Fonti consultate:**\n- [Docker Docs](https://docs.docker.com/engine/release-notes/)"),

        ("Qual è la differenza tra i core P-core e E-core di Intel?",
         "Intel Developer Zone (https://www.intel.com/content/www/us/en/developer/articles/technical/hybrid-architecture.html)",
         "I Performance-cores (P-cores) puntano alla massima frequenza e throughput single-thread, mentre gli Efficient-cores (E-cores, architettura Gracemont) massimizzano l'efficienza energetica multi-thread a basso consumo.",
         "I P-core sono progettati per carichi intensivi ad alte frequenze con Hyper-Threading, mentre gli E-core (come quelli sull'Intel N100) sono ottimizzati per l'efficienza energetica e carichi in background con consumi ridottissimi.\n\n**Fonti consultate:**\n- [Intel Developer Zone](https://www.intel.com/content/www/us/en/developer/articles/technical/hybrid-architecture.html)"),

        ("Come funziona la crittografia post-quantum ML-KEM?",
         "NIST Post-Quantum Cryptography (https://csrc.nist.gov/projects/post-quantum-cryptography)",
         "FIPS 203 standardizza ML-KEM (derivato da CRYSTALS-Kyber) come algoritmo di scambio chiavi basato su reticoli euclidei, resistente ad attacchi da computer quantistici.",
         "ML-KEM (FIPS 203) è lo standard crittografico post-quantistico del NIST basato sul problema matematico dei reticoli euclidei, progettato per proteggere le comunicazioni TLS contro futuri attacchi quantistici.\n\n**Fonti consultate:**\n- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)"),

        ("Che cos'è SQLite WAL mode e quali vantaggi offre?",
         "SQLite Consortium (https://www.sqlite.org/wal.html)",
         "Write-Ahead Logging (WAL) consente letture e scritture simultanee: i lettori non bloccano gli scrittori e gli scrittori non bloccano i lettori, riducendo drasticamente gli errori 'database is locked'.",
         "La modalità Write-Ahead Logging (WAL) separa le modifiche in un file di log temporaneo, permettendo letture e scritture concorrenti senza blocchi reciproci del database.\n\n**Fonti consultate:**\n- [SQLite Consortium](https://www.sqlite.org/wal.html)"),

        ("Qual è lo stato di approvazione della direttiva europea NIS2?",
         "European Cybersecurity Agency ENISA (https://www.enisa.europa.eu/topics/cybersecurity-policy/nis2)",
         "La direttiva NIS2 (UE 2022/2555) impone requisiti stringenti di gestione del rischio cyber e notifica incidenti entro 24 ore per enti essenziali e importanti.",
         "La direttiva NIS2 stabilisce misure obbligatorie di sicurezza per la supply chain digitale, gestione delle vulnerabilità e obbligo di prima notifica incidenti critici entro 24 ore.\n\n**Fonti consultate:**\n- [ENISA](https://www.enisa.europa.eu/topics/cybersecurity-policy/nis2)")
    ]

    samples = []
    # Replicate with realistic prompt phrasing
    queries_templates = [
        "Cerca su internet {topic} e fammi un riassunto",
        "Cerca novità su {topic}",
        "Trovami informazioni aggiornate su: {topic}",
        "Cosa dicono le fonti web su {topic}?"
    ]

    while len(samples) < target_count:
        q_text, src, snippet, answer = random.choice(topics)
        tmpl = random.choice(queries_templates)
        user_query = tmpl.format(topic=q_text.replace("?", "").replace("Quali sono le ", "").replace("Cosa è ", ""))

        rag_system = (
            f"Sei C-3PO, assistente di ricerca intelligente per HOME-SERVER.\n"
            f"Rispondi in modo brillante, chiaro e accurato in italiano basandoti ESCLUSIVAMENTE sulle seguenti fonti web.\n"
            f"Concludi sempre con le fonti consultate nel formato:\n"
            f"**Fonti consultate:**\n- [Titolo Fonte](URL)\n\n"
            f"[RISULTATI RICERCA WEB]:\n--- FONTE: {src} ---\n{snippet}"
        )

        samples.append({
            "messages": [
                {"role": "system", "content": rag_system},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": answer}
            ]
        })

    print(f"    Completato Pillar 2: {len(samples)} campioni")
    return samples

# =========================================================================
# PILLAR 3: Technical PDF & Markdown Reports (15%)
# =========================================================================
def build_pdf_report_samples(target_count: int = 225) -> List[Dict[str, Any]]:
    print(f"--> [Pillar 3: 15%] Generazione {target_count} campioni di Report PDF / Markdown...")
    report_blueprints = [
        ("Best practice di sicurezza Docker",
         "# SICUREZZA DEI CONTAINER DOCKER\n\n"
         "## 1. Obiettivi e Architettura\n"
         "L'isolamento dei container richiede una configurazione hardening dell'host Linux e dei singoli demoni.\n\n"
         "## 2. Best Practice Operative\n"
         "* Eseguire i processi con utenti non-root (`USER appuser`).\n"
         "* Montare il filesystem di root in sola lettura (`read_only: true`).\n"
         "* Rimuovere i permessi Linux non necessari con `cap_drop: [ALL]`.\n"
         "* Impostare limiti ferrei su memoria (`mem_limit`) e CPU per prevenire attacchi DoS.\n\n"
         "## 3. Raccomandazioni di Sicurezza\n"
         "Mantenere sempre aggiornata l'immagine base e scansionare regolarmente le dipendenze software."),

        ("Architettura Home-Server Intel N100",
         "# ARCHITETTURA HOME-SERVER INTEL N100\n\n"
         "## 1. Sintesi Infrastrutturale\n"
         "Sistema server a bassissimo consumo energetico basato su CPU Intel Alder Lake-N100 a 4 E-core e 12GB RAM.\n\n"
         "## 2. Configurazione dei Servizi\n"
         "* Reverse Proxy Nginx con SSL/TLS automatico e gestione WebSocket/SSE.\n"
         "* 23 Microservizi containerizzati isolati in reti bridge interne.\n"
         "* Storage NVMe ad alta velocità con database SQLite WAL per zero overhead.\n\n"
         "## 3. Parametri di Efficienza\n"
         "Consumo medio di 6W in idle, temperature operative inferiori a 55°C e affidabilità 24/7."),

        ("Protocollo di risposta agli incidenti SOC",
         "# PROTOCOLLO DI RISPOSTA AGLI INCIDENTI SOC\n\n"
         "## 1. Fase di Rilevamento e Triage\n"
         "Identificazione tempestiva di attività anomale tramite telemetria host e log Nginx.\n\n"
         "## 2. Azioni di Contenimento\n"
         "* Isolamento immediato dell'indirizzo IP malevolo con `fail2ban` o `iptables`.\n"
         "* Revoca delle sessioni attive e rotazione token crittografici.\n"
         "* Ispezione dei descrittori di rete e dei container compromessi.\n\n"
         "## 3. Eradicazione e Ripristino\n"
         "Patching della vulnerabilità sorgente e audit di integrità prima della rimessa in produzione.")
    ]

    samples = []
    user_requests = [
        "Crea un documento PDF su {topic}",
        "Genera una guida tecnica in PDF per {topic}",
        "Redigi un report formale PDF sul tema {topic}",
        "Prepara un documento Markdown strutturato su {topic}"
    ]

    while len(samples) < target_count:
        topic_title, md_content = random.choice(report_blueprints)
        tmpl = random.choice(user_requests)
        user_prompt = tmpl.format(topic=topic_title)

        doc_system = (
            f"Sei un redattore tecnico di sistemi. Redigi una guida sintetica ad alta densità (150-250 parole) su: \"{topic_title}\".\n"
            f"Struttura formale:\n# {topic_title.upper()}\n## 1. Obiettivi\n## 2. Best Practice (usa elenchi puntati con '*')\n## 3. Raccomandazioni\n"
            f"Italiano autorevole e chirurgico. Inizia subito con il Titolo (#)."
        )

        samples.append({
            "messages": [
                {"role": "system", "content": doc_system},
                {"role": "user", "content": f"Redigi la guida tecnica su: {topic_title}"},
                {"role": "assistant", "content": md_content}
            ]
        })

    print(f"    Completato Pillar 3: {len(samples)} campioni")
    return samples

# =========================================================================
# PILLAR 4: Coding, Linux Sysadmin & Cybersecurity SOC (15%)
# =========================================================================
def build_coding_and_soc_samples(target_count: int = 225, seed: int = 42) -> List[Dict[str, Any]]:
    print(f"--> [Pillar 4: 15%] Generazione {target_count} campioni di Coding & Cybersecurity SOC...")
    
    # 1. Specialist SOC & Linux Diagnostics
    soc_cases = [
        ("Come isolare un IP sospetto 194.26.29.112 con iptables?",
         "Esegui: `sudo iptables -I INPUT -s 194.26.29.112 -j DROP`. Per renderlo persistente: `sudo netfilter-persistent save`."),
        
        ("Come monitorare la RAM occupata da ciascun container Docker in tempo reale?",
         "Esegui: `docker stats --format 'table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}'`."),

        ("Come analizzare le connessioni TCP attive su porta 443?",
         "Esegui: `ss -tuanp | grep :443` oppure `netstat -tulnp | grep 443`."),

        ("Come verificare se la CPU Intel N100 sta applicando thermal throttling?",
         "Controlla i registri MSR del kernel: `dmesg -T | grep -i 'thermal\\|throttle'` e controlla la temperatura con `sensors`."),

        ("Come riavviare solo il reverse-proxy Nginx preservando le sessioni?",
         "Esegui: `docker compose exec reverse-proxy nginx -s reload` per ricaricare la configurazione senza terminare le connessioni aperte.")
    ]

    samples = []
    for q, a in soc_cases:
        for _ in range(15):
            samples.append({
                "messages": [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

    # 2. Magicoder programming samples (Python / C / Bash)
    remaining = max(0, target_count - len(samples))
    if remaining > 0:
        ds_code = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        items = list(ds_code)
        random.seed(seed)
        random.shuffle(items)

        for it in items:
            inst = (it.get("problem") or it.get("instruction") or "").strip()
            sol = (it.get("solution") or it.get("response") or "").strip()
            if not inst or not sol:
                continue
            if len(sol) < 30 or len(sol) > 1000:
                continue

            samples.append({
                "messages": [
                    {"role": "system", "content": "Sei un programmatore esperto. Fornisci soluzioni concise, pulite ed efficienti."},
                    {"role": "user", "content": inst},
                    {"role": "assistant", "content": clean_fluff(sol)}
                ]
            })
            if len(samples) >= target_count:
                break

    print(f"    Completato Pillar 4: {len(samples)} campioni")
    return samples

# =========================================================================
# MAIN DATASET ASSEMBLER
# =========================================================================
def main():
    output_dir = "finetune/data"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("MINICPM-2B: CREAZIONE DATASET MULTI-TASK BILANCIATO")
    print("=" * 65)

    p1 = build_italian_reasoning_samples(target_count=600)  # 40%
    p2 = build_web_rag_samples(target_count=450)            # 30%
    p3 = build_pdf_report_samples(target_count=225)         # 15%
    p4 = build_coding_and_soc_samples(target_count=225)     # 15%

    all_data = p1 + p2 + p3 + p4
    random.seed(42)
    random.shuffle(all_data)

    total = len(all_data)
    eval_count = max(10, int(total * 0.05))
    train_data = all_data[:-eval_count]
    eval_data = all_data[-eval_count:]

    train_path = os.path.join(output_dir, "train_multitask.json")
    eval_path = os.path.join(output_dir, "eval_multitask.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("=" * 65)
    print(f"Dataset Multi-Task generato con successo!")
    print(f"Totale campioni: {total}")
    print(f"Train split (95%): {len(train_data)} -> {train_path}")
    print(f"Eval split  (5%):  {len(eval_data)} -> {eval_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
