# -*- coding: utf-8 -*-
"""
Dataset multi-task per MiniCPM-2B (RTX 2080 -> Intel N100), allineato al runtime.

Regole che il dataset precedente violava:
- Nessuna risposta ripetuta: 6 risposte copiate ~75 volte (un terzo dei dati)
  insegnavano al modello a ripetere frasi fisse a prescindere dalla domanda.
- Stessi system prompt del runtime: letti da home-server/ai-soc-app/lib/prompts.json.
- Stesso formato del turno user del runtime: [TELEMETRIA] / [FONTI] / [DOMANDA]
  come in prompt-builder.js, con numeri diversi in ogni esempio, cosi' il
  modello impara a leggere i dati invece di ricordarli.
- Il modello non scrive mai link: nel RAG cita [1], [2], [3]; gli URL li
  aggiunge il codice.

Composizione (default, ~4000 esempi):
  40%  chat in italiano (alpaca-italian-cleaned + dataset extra opzionale), 15% a 2 turni
  20%  RAG con 3 fonti numerate (SQuAD-it, fallback su alpaca), 10% senza risposta nelle fonti
  15%  coding (Magicoder-OSS-Instruct)
  12%  SOC/Linux con telemetria sintetica chiave=valore
  10%  documenti Markdown strutturati (## Sommario, ##, ###, elenchi)
   3%  saluti e identita' (varianti uniche)

Uso:
  python finetune/prepare_multitask_dataset.py --total 4000
  python finetune/prepare_multitask_dataset.py --extra-it <dataset HF con instruction/input/output>
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = REPO_ROOT / "home-server" / "ai-soc-app" / "lib" / "prompts.json"
PROMPTS = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))

MAX_CHARS = 3200          # ~1000 token: oltre, il campione verrebbe troncato in training

PREAMBLE_REGEX = re.compile(
    r"^(?:Certo,?|Certamente,?!?|Ecco,?|Ecco a te,?|Di seguito,?|Sicuramente,?|"
    r"In qualità di assistente,?|Come assistente IA,?|Sì,? posso aiutarti\.?|Volentieri,?)\s*",
    re.IGNORECASE,
)


def clean_fluff(text: str) -> str:
    cleaned = PREAMBLE_REGEX.sub("", text.strip()).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def msg(system_key: str, *turns: str) -> Dict:
    """turns = user, assistant, user, assistant, ..."""
    messages = [{"role": "system", "content": PROMPTS[system_key]}]
    for i, t in enumerate(turns):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return {"task": system_key, "messages": messages}


def user_turn(question: str, telemetry: Optional[str] = None, sources: Optional[str] = None) -> str:
    """Stesso formato di buildUserTurn() in prompt-builder.js."""
    parts = []
    if telemetry:
        parts.append(f"[TELEMETRIA]\n{telemetry}")
    if sources:
        parts.append(f"[FONTI]\n{sources}")
    if not parts:
        return question.strip()
    parts.append(f"[DOMANDA]\n{question.strip()}")
    return "\n\n".join(parts)


def load_hf(name: str, split: str = "train", **kwargs):
    try:
        from datasets import load_dataset
        return list(load_dataset(name, split=split, **kwargs))
    except Exception as exc:  # rete assente, dataset rinominato, ...
        print(f"    [!] Dataset {name} non disponibile: {exc}")
        return []


def first_sentences(text: str, n: int = 2, max_chars: int = 400) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sents[:n]).strip()
    return out[:max_chars].rsplit(" ", 1)[0] + "…" if len(out) > max_chars else out


# =========================================================================
# 1. Chat in italiano
# =========================================================================
def load_italian_pool(extra: Optional[str], seed: int) -> List[Dict]:
    pool = []
    for name in ["saillab/alpaca-italian-cleaned"] + ([extra] if extra else []):
        for it in load_hf(name):
            instr = (it.get("instruction") or "").strip()
            inp = (it.get("input") or "").strip()
            out = (it.get("output") or "").strip()
            if not instr or not out or len(out) < 30 or len(out) > 1500:
                continue
            if re.search(r"https?://|!\[", out):
                continue
            pool.append({"q": f"{instr}\n{inp}".strip() if inp else instr, "a": clean_fluff(out), "instr": instr, "raw": out})
    random.Random(seed).shuffle(pool)
    print(f"    pool italiano: {len(pool)} esempi utilizzabili")
    return pool


def build_chat(pool: List[Dict], count: int, multi_turn_ratio: float = 0.15) -> List[Dict]:
    out = []
    i = 0
    while len(out) < count and i + 1 < len(pool):
        if random.random() < multi_turn_ratio:
            a, b = pool[i], pool[i + 1]
            out.append(msg("chat", a["q"], a["a"], b["q"], b["a"]))
            i += 2
        else:
            out.append(msg("chat", pool[i]["q"], pool[i]["a"]))
            i += 1
    return out


# =========================================================================
# 2. RAG con fonti numerate
# =========================================================================
def build_rag(pool: List[Dict], count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for it in load_hf("crux82/squad_it"):
        ctx = (it.get("context") or "").strip()
        q = (it.get("question") or "").strip()
        answers = (it.get("answers") or {}).get("text") or []
        if ctx and q and answers and 200 < len(ctx) < 1200:
            ans = answers[0]
            sent = next((s for s in re.split(r"(?<=[.!?])\s+", ctx) if ans in s), ans)
            items.append({"title": (it.get("title") or "Fonte").replace("_", " "), "text": ctx, "q": q, "a": sent})
    if not items:  # fallback: le risposte alpaca diventano fonti
        for it in pool:
            if 250 < len(it["raw"]) < 1100:
                items.append({"title": it["instr"][:60], "text": it["raw"], "q": it["q"], "a": first_sentences(it["raw"], 2)})
    rng.shuffle(items)
    print(f"    fonti RAG disponibili: {len(items)}")

    out = []
    for idx, item in enumerate(items):
        if len(out) >= count:
            break
        distractors = rng.sample(items, 3)
        distractors = [d for d in distractors if d is not item][:2]
        unanswerable = rng.random() < 0.10
        docs = distractors + ([] if unanswerable else [item])
        if unanswerable:
            docs.append(rng.choice(items))
        rng.shuffle(docs)
        sources = "\n\n".join(f"[{n}] {d['title']}\n{d['text'][:900]}" for n, d in enumerate(docs, 1))
        if unanswerable and item not in docs:
            answer = "Le fonti fornite non contengono informazioni sufficienti per rispondere a questa domanda."
        else:
            n = docs.index(item) + 1
            answer = f"{item['a'].rstrip('.')} [{n}]."
        out.append(msg("rag", user_turn(item["q"], sources=sources), answer))
    return out


# =========================================================================
# 3. Coding
# =========================================================================
def build_code(count: int, seed: int) -> List[Dict]:
    rows = load_hf("ise-uiuc/Magicoder-OSS-Instruct-75K")
    random.Random(seed).shuffle(rows)
    out = []
    for it in rows:
        q = (it.get("problem") or "").strip()
        a = (it.get("solution") or "").strip()
        if q and a and 40 < len(a) < 1800 and len(q) < 1400:
            out.append(msg("chat", q, clean_fluff(a)))
        if len(out) >= count:
            break
    return out


# =========================================================================
# 4. SOC con telemetria sintetica (stesso formato di compactTelemetry())
# =========================================================================
CONTAINERS = ["reverse-proxy", "home-portal", "web-app", "chat-app", "mail-app", "cloud-app", "vault-app",
              "calendar-app", "meal-app", "movie-app", "smart-home-app", "backup-app", "ai-soc-app",
              "minicpm2b-n100", "docker-proxy", "pdfcraft-app", "wordcraft-app", "vademecum-app", "sign-app"]


def fake_telemetry(rng: random.Random) -> Dict:
    total_ram = 11.2
    pct = rng.choice([rng.randint(18, 60), rng.randint(60, 80), rng.randint(80, 97)])
    temp = rng.choice([rng.randint(35, 58), rng.randint(58, 72), rng.randint(72, 92)])
    total_c = rng.randint(18, 24)
    bad = rng.sample(CONTAINERS, rng.choice([0, 0, 0, 1, 2]))
    states = {name: rng.choice(["unhealthy", "exited", "restarting"]) for name in bad}
    ssh_failed = rng.choice([0, rng.randint(1, 15), rng.randint(40, 400)])
    suspects = 0 if ssh_failed < 40 else rng.randint(1, 6)
    n5xx = rng.choice([0, rng.randint(1, 9), rng.randint(11, 120)])
    issues = []
    if any(s == "unhealthy" for s in states.values()):
        issues.append(f"{sum(s == 'unhealthy' for s in states.values())} container in stato UNHEALTHY")
    if pct > 88:
        issues.append(f"Pressione RAM elevata ({pct}%)")
    if suspects:
        issues.append(f"{suspects} IP sospetti di brute-force SSH")
    if n5xx > 10:
        issues.append(f"Spike di errori 5xx ({n5xx}) su Nginx")
    score = max(20, 100 - 15 * sum(s == "unhealthy" for s in states.values()) - (10 if pct > 88 else 0) - 5 * suspects - (8 if n5xx > 10 else 0))
    level = "CRITICO" if score < 60 else "ATTENZIONE" if score < 80 else "BUONO" if score < 95 else "ECCELLENTE"
    load = [round(rng.uniform(0.1, 4.5), 2) for _ in range(3)]
    return {"pct": pct, "used": round(total_ram * pct / 100, 1), "total": total_ram, "temp": temp, "load": load,
            "total_c": total_c, "running": total_c - sum(s != "unhealthy" for s in states.values()),
            "states": states, "ssh_failed": ssh_failed, "suspects": suspects, "n4xx": rng.randint(0, 80),
            "n5xx": n5xx, "issues": issues, "score": score, "level": level,
            "missing": rng.random() < 0.08}


def compact(t: Dict, rng: random.Random) -> str:
    probs = ", ".join(f"{k}:{v}" for k, v in t["states"].items())
    ram = "n/d/n/d (n/d%)" if t["missing"] else f"{t['used']} GB/{t['total']} GB ({t['pct']}%)"
    return "\n".join([
        f"ora=2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
        f"score={t['score']}/100 ({t['level']})",
        f"cpu_temp={t['temp']}C load={'/'.join(f'{v:.2f}' for v in t['load'])}",
        f"ram={ram}",
        f"container={t['running']}/{t['total_c']} unhealthy={sum(v == 'unhealthy' for v in t['states'].values())}" + (f" problemi=[{probs}]" if probs else ""),
        f"ssh_falliti={t['ssh_failed']} ip_sospetti={t['suspects']}",
        f"nginx_4xx={t['n4xx']} nginx_5xx={t['n5xx']}",
        f"criticita={'; '.join(t['issues']) if t['issues'] else 'nessuna'}",
    ])


def soc_answer(kind: str, t: Dict) -> str:
    if kind == "ram":
        if t["missing"]:
            return "Il dato sulla RAM non è presente nella telemetria. Puoi verificarlo direttamente con `free -h`."
        if t["pct"] < 75:
            return f"La RAM in uso è {t['used']} GB su {t['total']} GB ({t['pct']}%): un livello normale, c'è margine sufficiente."
        if t["pct"] < 88:
            return f"La RAM in uso è {t['used']} GB su {t['total']} GB ({t['pct']}%): è alta ma non critica. Controlla i container più pesanti con `docker stats --no-stream`."
        return (f"La RAM è al {t['pct']}% ({t['used']} GB su {t['total']} GB): pressione elevata. "
                "Individua i processi più pesanti con `docker stats --no-stream` e `ps aux --sort=-%mem | head`, poi valuta limiti `mem_limit` sui container.")
    if kind == "temp":
        if t["temp"] < 65:
            return f"La CPU è a {t['temp']}°C: temperatura normale per l'Intel N100."
        if t["temp"] < 80:
            return f"La CPU è a {t['temp']}°C: valore alto ma sotto la soglia critica. Verifica il carico ({'/'.join(map(str, t['load']))}) e la ventilazione con `sensors`."
        return f"La CPU è a {t['temp']}°C: temperatura critica. Controlla subito ventilazione e processi con `top`, e i log del kernel con `dmesg -T | grep -i thermal`."
    if kind == "containers":
        if not t["states"]:
            return f"Tutti i container sono in esecuzione ({t['running']}/{t['total_c']}) e nessuno risulta unhealthy."
        lines = [f"Ci sono container da controllare ({t['running']}/{t['total_c']} in esecuzione):"]
        for name, state in t["states"].items():
            lines.append(f"- `{name}`: {state}. Controlla con `docker logs --tail 50 {name}`.")
        return "\n".join(lines)
    if kind == "ssh":
        if t["suspects"]:
            return (f"Sì: {t['ssh_failed']} accessi SSH falliti e {t['suspects']} IP sospetti di brute-force. "
                    "Verifica i ban con `sudo fail2ban-client status sshd` e controlla `/var/log/auth.log`.")
        if t["ssh_failed"]:
            return f"Ci sono {t['ssh_failed']} accessi SSH falliti ma nessun IP sospetto di brute-force: attività di fondo normale."
        return "Nessun accesso SSH fallito e nessun IP sospetto nella telemetria attuale."
    if kind == "nginx":
        if t["n5xx"] > 10:
            return f"Nginx registra {t['n5xx']} errori 5xx: un servizio a monte non risponde. Controlla `docker logs --tail 100 reverse-proxy` e lo stato dei container."
        return f"Nginx è regolare: {t['n5xx']} errori 5xx e {t['n4xx']} errori 4xx (questi ultimi sono in genere richieste di client)."
    # stato generale
    if not t["issues"]:
        return f"Il server è in buono stato: punteggio {t['score']}/100 ({t['level']}), {t['running']}/{t['total_c']} container attivi, CPU a {t['temp']}°C."
    return f"Punteggio {t['score']}/100 ({t['level']}). Criticità: " + "; ".join(t["issues"]) + "."


SOC_QUESTIONS = {
    "ram": ["quanta RAM sta usando il server?", "la memoria è piena?", "come siamo messi con la RAM?", "consumo di memoria attuale?"],
    "temp": ["la temperatura della CPU è normale?", "il processore sta scaldando troppo?", "che temperatura ha la CPU adesso?"],
    "containers": ["ci sono container con problemi?", "tutti i container sono attivi?", "quali container sono fermi o unhealthy?"],
    "ssh": ["ci sono attacchi SSH in corso?", "qualcuno sta provando ad accedere via ssh?", "ci sono tentativi di intrusione?"],
    "nginx": ["nginx sta dando errori?", "il reverse proxy ha problemi?", "ci sono errori 5xx?"],
    "status": ["come sta il server?", "fammi un riepilogo dello stato del sistema", "ci sono criticità sul server?"],
}


def build_soc(count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    # Le risposte "tutto regolare" sono poche e verrebbero scartate dal dedup:
    # si genera finche' non ci sono `count` risposte diverse
    for _ in range(count * 20):
        if len(out) >= count:
            break
        kind = rng.choice(list(SOC_QUESTIONS))
        t = fake_telemetry(rng)
        answer = soc_answer(kind, t)
        if answer in seen:
            continue
        seen.add(answer)
        q = rng.choice(SOC_QUESTIONS[kind])
        out.append(msg("soc", user_turn(q, telemetry=compact(t, rng)), answer))
    return out


# =========================================================================
# 5. Documenti Markdown (stesso turno user di handleDocument())
# =========================================================================
def to_document(raw: str) -> Optional[str]:
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    list_items = [re.sub(r"^(?:\d+[.)]|[-*•])\s*", "", l) for l in lines if re.match(r"^(?:\d+[.)]|[-*•])\s+", l)]
    prose = [l for l in lines if not re.match(r"^(?:\d+[.)]|[-*•])\s+", l)]
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(prose))
    if len(list_items) < 3 and len(sentences) < 5:
        return None
    summary = " ".join(sentences[:2]).strip()
    body = list_items or [s for s in sentences[2:] if len(s) > 25]
    if len(body) < 3:
        return None
    half = max(2, len(body) // 2)
    doc = [f"## Sommario\n{summary}", "## Punti principali", "\n".join(f"- {b}" for b in body[:half])]
    if body[half:]:
        doc += ["### Dettagli", "\n".join(f"- {b}" for b in body[half:])]
    return "\n\n".join(doc)


def build_docs(pool: List[Dict], count: int) -> List[Dict]:
    out = []
    for it in pool:
        if len(it["raw"]) < 450:
            continue
        doc = to_document(it["raw"])
        if not doc:
            continue
        topic = it["instr"].rstrip("?.!")
        request = f"Crea un documento PDF su: {topic[0].lower() + topic[1:]}"
        turn = user_turn(f"Argomento: {topic}\nRichiesta: {request}\nLunghezza: 350-550 parole. Usa una tabella se ci sono dati da confrontare.")
        out.append(msg("doc", turn, doc))
        if len(out) >= count:
            break
    return out


# =========================================================================
# 6. Saluti e identita' (nessuna risposta ripetuta)
# =========================================================================
GREETINGS = [
    (["come stai?", "come stai ?", "ciao, come va?"], ["Sto bene, grazie! Tu come stai? Dimmi pure se posso aiutarti con qualcosa.",
                                                      "Tutto bene, grazie per averlo chiesto. Di cosa hai bisogno oggi?",
                                                      "Bene, grazie! Sono pronto ad aiutarti: chiedimi pure."]),
    (["ciao", "buongiorno", "salve"], ["Ciao! Come posso aiutarti?", "Buongiorno! Dimmi pure di cosa hai bisogno.",
                                       "Salve! Sono C-3PO, l'assistente di HOME-SERVER. In cosa posso esserti utile?"]),
    (["grazie", "grazie mille", "perfetto, grazie"], ["Prego! Se ti serve altro, sono qui.", "Figurati, è stato un piacere.",
                                                      "Di nulla! Chiedimi pure se hai altre domande."]),
    (["chi sei?", "cosa sai fare?"], ["Sono C-3PO, l'assistente di HOME-SERVER: rispondo a domande, scrivo codice, controllo lo stato del server, cerco sul web, creo documenti PDF e immagini.",
                                      "Sono C-3PO. Posso conversare, aiutarti con la programmazione, analizzare lo stato del server, fare ricerche web con fonti e preparare documenti."]),
    (["buonanotte", "a domani"], ["Buonanotte! A domani.", "A presto, buona serata!"]),
]


def build_greetings() -> List[Dict]:
    out = []
    for questions, answers in GREETINGS:
        for q in questions:
            for a in answers:
                out.append(msg("chat", q, a))
    return out


# =========================================================================
# Assemblaggio
# =========================================================================
def dedup(samples: List[Dict]) -> List[Dict]:
    seen_pairs, seen_answers, out = set(), {}, []
    for s in samples:
        text = json.dumps(s["messages"], ensure_ascii=False)
        if len(text) > MAX_CHARS + 600:
            continue
        pair = hashlib.sha1(text.encode("utf-8")).hexdigest()
        answer = hashlib.sha1(s["messages"][-1]["content"].strip().lower().encode("utf-8")).hexdigest()
        # Saluti: max 2 copie per risposta; tutto il resto: risposta unica
        limit = 2 if s["task"] == "chat" and len(s["messages"][-1]["content"]) < 200 else 1
        if pair in seen_pairs or seen_answers.get(answer, 0) >= limit:
            continue
        seen_pairs.add(pair)
        seen_answers[answer] = seen_answers.get(answer, 0) + 1
        out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total", type=int, default=4000)
    parser.add_argument("--extra-it", default=None, help="dataset HF extra in italiano (colonne instruction/input/output)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "finetune" / "data"))
    args = parser.parse_args()

    random.seed(args.seed)
    n = args.total
    pool = load_italian_pool(args.extra_it, args.seed)
    # Pool diviso per pillar: nessun esempio alpaca riusato in due compiti
    third = len(pool) // 3
    chat_pool, rag_pool, doc_pool = pool[:third * 2], pool[third * 2:third * 2 + third // 2], pool[third * 2 + third // 2:]

    parts = {
        "chat": build_chat(chat_pool, int(n * 0.40)),
        "rag": build_rag(rag_pool, int(n * 0.20), args.seed),
        "code": build_code(int(n * 0.15), args.seed),
        "soc": build_soc(int(n * 0.12), args.seed),
        "doc": build_docs(doc_pool, int(n * 0.10)),
        "greetings": build_greetings(),
    }
    for k, v in parts.items():
        print(f"--> {k:9s}: {len(v)} esempi")

    data = dedup([s for v in parts.values() for s in v])
    random.shuffle(data)
    eval_count = max(20, int(len(data) * 0.04))
    train, evals = data[:-eval_count], data[-eval_count:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train_multitask.json", train), ("eval_multitask.json", evals)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)

    uniq = len({s["messages"][-1]["content"] for s in data})
    print(f"Totale dopo deduplica: {len(data)} (risposte uniche: {uniq}) -> train {len(train)}, eval {len(evals)}")


if __name__ == "__main__":
    main()
