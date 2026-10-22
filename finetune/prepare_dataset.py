# -*- coding: utf-8 -*-
"""
Script per il download, la normalizzazione e la preparazione di dataset ad alto ragionamento
per il Fine-Tuning QLoRA di MiniCPM-2B.
Supporta:
- Dataset Hugging Face (formati ShareGPT e Alpaca)
- File JSON/JSONL locali
- Pulizia rigorosa (rimozione turni vuoti, validazione alternanza ruoli)
- Formattazione standard ChatML: [{"messages": [{"role": "...", "content": "..."}]}]
- Split 95% Training / 5% Validation (train_custom.json / eval_custom.json)
"""

import os
import sys
import json
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


DEFAULT_SYSTEM_PROMPT = (
    "Sei un assistente AI diretto, preciso ed esperto in informatica, logica e programmazione. "
    "Rispondi in modo tecnico, analitico e conciso senza preamboli inutili."
)

# Dataset verificati ad accesso libero per logica, coding e lingua italiana
PRESET_DATASETS = {
    "alpaca-it": {
        "hf_id": "FreedomIntelligence/alpaca-gpt4-italian",
        "type": "sharegpt",
        "desc": "Dataset di istruzioni GPT-4 ad alto ragionamento in lingua italiana (formato ShareGPT)."
    },
    "alpaca-it-cleaned": {
        "hf_id": "saillab/alpaca-italian-cleaned",
        "type": "alpaca",
        "desc": "Dataset curato e pulito di istruzioni in lingua italiana (formato Alpaca)."
    },
    "python-code": {
        "hf_id": "iamtarun/python_code_instructions_18k_alpaca",
        "type": "alpaca",
        "desc": "Dataset specialistico per programmazione e problem-solving Python (formato Alpaca)."
    }
}


def normalize_sharegpt(row: Dict[str, Any], system_prompt: str) -> Optional[List[Dict[str, str]]]:
    """Converte formato ShareGPT (conversations con from/value o messages con role/content)."""
    turns = row.get("conversations") or row.get("messages") or []
    if not turns:
        return None

    messages = [{"role": "system", "content": system_prompt}]
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "chatgpt": "assistant",
        "assistant": "assistant",
        "system": "system"
    }

    for turn in turns:
        raw_role = turn.get("from") or turn.get("role", "")
        role = role_map.get(raw_role.lower(), raw_role.lower())
        content = (turn.get("value") or turn.get("content") or "").strip()

        if not content:
            continue

        if role == "system":
            messages[0]["content"] = content
        elif role in ["user", "assistant"]:
            messages.append({"role": role, "content": content})

    return validate_chatml_messages(messages)


def normalize_alpaca(row: Dict[str, Any], system_prompt: str) -> Optional[List[Dict[str, str]]]:
    """Converte formato Alpaca (instruction, input opzionale, output)."""
    instruction = (row.get("instruction") or "").strip()
    input_text = (row.get("input") or "").strip()
    output_text = (row.get("output") or row.get("response") or "").strip()

    if not instruction or not output_text:
        return None

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n[Dati/Contesto]:\n{input_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text}
    ]

    return validate_chatml_messages(messages)


def validate_chatml_messages(messages: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """
    Verifica che la conversazione sia valida:
    - Almeno 1 turno user e 1 assistant
    - Nessun messaggio vuoto
    - L'ultimo messaggio deve essere dell'assistant
    - Alternanza corretta user -> assistant
    """
    if len(messages) < 3:
        return None

    cleaned = [messages[0]]
    expected_role = "user"

    for msg in messages[1:]:
        role = msg["role"]
        content = msg["content"].strip()
        if not content or role not in ["user", "assistant"]:
            continue

        if role == expected_role:
            cleaned.append({"role": role, "content": content})
            expected_role = "assistant" if role == "user" else "user"
        elif role == cleaned[-1]["role"]:
            cleaned[-1]["content"] += "\n\n" + content

    if len(cleaned) < 3 or cleaned[-1]["role"] != "assistant":
        return None

    return cleaned


def prepare_dataset(
    source: str = "alpaca-it-cleaned",
    dataset_type: str = "auto",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    output_dir: str = "finetune/data",
    max_samples: Optional[int] = None,
    eval_split_ratio: float = 0.05,
    seed: int = 42
):
    print("=" * 65)
    print("MiniCPM-2B: Preparazione Dataset ChatML ad Alto Ragionamento")
    print("=" * 65)
    print(f"Sorgente:            {source}")
    print(f"Tipo formato:        {dataset_type}")
    print(f"Directory di output: {output_dir}")
    print(f"Split Train / Eval:  {int((1 - eval_split_ratio)*100)}% / {int(eval_split_ratio*100)}%")

    # Rilevamento preset o percorso
    if source in PRESET_DATASETS:
        preset = PRESET_DATASETS[source]
        print(f"--> Utilizzo preset: {source} ({preset['desc']})")
        hf_source = preset["hf_id"]
        if dataset_type == "auto":
            dataset_type = preset["type"]
    else:
        hf_source = source

    raw_items = []

    # Caricamento da file locale o da Hugging Face
    if os.path.exists(hf_source):
        print(f"--> Caricamento file locale: {hf_source}")
        with open(hf_source, "r", encoding="utf-8") as f:
            if hf_source.endswith(".jsonl"):
                raw_items = [json.loads(line) for line in f if line.strip()]
            else:
                raw_items = json.load(f)
    else:
        if load_dataset is None:
            raise ImportError("La libreria 'datasets' è necessaria. Esegui: pip install datasets")
        print(f"--> Download dataset da Hugging Face: {hf_source} ...")
        ds = load_dataset(hf_source, split="train")
        raw_items = [item for item in ds]

    print(f"--> Campioni grezzi caricati: {len(raw_items)}")

    # Rilevamento automatico del formato se non specificato
    if dataset_type == "auto" and raw_items:
        first = raw_items[0]
        if "conversations" in first or "messages" in first:
            dataset_type = "sharegpt"
        elif "instruction" in first and "output" in first:
            dataset_type = "alpaca"
        else:
            dataset_type = "sharegpt"
        print(f"--> Formato rilevato automaticamente: {dataset_type}")

    # Normalizzazione e filtraggio
    processed_conversations = []
    for row in raw_items:
        if dataset_type == "sharegpt":
            norm = normalize_sharegpt(row, system_prompt)
        elif dataset_type == "alpaca":
            norm = normalize_alpaca(row, system_prompt)
        else:
            norm = normalize_sharegpt(row, system_prompt)

        if norm:
            processed_conversations.append({"messages": norm})

    print(f"--> Campioni validati e formattati ChatML: {len(processed_conversations)}")

    # Limitazione campioni opzionale
    random.seed(seed)
    random.shuffle(processed_conversations)
    if max_samples and max_samples < len(processed_conversations):
        processed_conversations = processed_conversations[:max_samples]
        print(f"--> Dataset limitato a: {max_samples} campioni richiesti")

    # Split Train / Eval (95% / 5%)
    num_eval = max(1, int(len(processed_conversations) * eval_split_ratio))
    eval_data = processed_conversations[:num_eval]
    train_data = processed_conversations[num_eval:]

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train_custom.json")
    eval_file = os.path.join(output_dir, "eval_custom.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("SUCCESS: Dataset generati con successo!")
    print(f"--> Training Set:   {train_file} ({len(train_data)} esempi)")
    print(f"--> Validation Set: {eval_file} ({len(eval_data)} esempi)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset ChatML per MiniCPM-2B")
    parser.add_argument(
        "--source",
        type=str,
        default="alpaca-it-cleaned",
        help="Preset ('alpaca-it-cleaned', 'alpaca-it', 'python-code'), ID Hugging Face, o percorso JSON/JSONL"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "sharegpt", "alpaca"],
        help="Tipo di formato dei dati"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Numero massimo di campioni da estrarre (default 5000)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt tecnico da applicare a tutte le conversazioni"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune/data",
        help="Directory dove salvare train_custom.json ed eval_custom.json"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="Percentuale del dataset per la validazione (default 0.05 = 5%)"
    )

    args = parser.parse_args()
    prepare_dataset(
        source=args.source,
        dataset_type=args.type,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        eval_split_ratio=args.eval_ratio
    )
