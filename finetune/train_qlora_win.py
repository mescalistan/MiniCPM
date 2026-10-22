# -*- coding: utf-8 -*-
"""
Script ottimizzato per Fine-Tuning QLoRA di MiniCPM-2B su Windows con GPU NVIDIA (RTX 2080 8GB).
Caratteristiche:
- Quantizzazione 4-bit NF4 con BitsAndBytes (VRAM < 5 GB)
- Targeting di tutti i moduli lineari (q, k, v, o, gate, up, down)
- Precisione di calcolo FP16 ottimale per Tensor Core Turing (sm_75)
- Formato MiniCPM nativo (<用户>/<AI>) identico al template di llama-server, anche multi-turno
- Loss solo sui token delle risposte (maschera sugli offset dei caratteri)
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="openbmb/MiniCPM-2B-dpo-fp16",
        metadata={"help": "Path locale o ID Hugging Face del modello base ufficiale (MiniCPM-2B-dpo-fp16)."}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="finetune/data/train_github.json",
        metadata={"help": "Percorso del file JSON di training (formato ChatML/MiniCPM)."}
    )
    eval_data_path: Optional[str] = field(
        default="finetune/data/eval_github.json",
        metadata={"help": "Percorso del file JSON di validazione."}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Lunghezza massima della sequenza (troncamento/padding)."}
    )
    lora_r: int = field(default=16, metadata={"help": "Rank LoRA."})
    lora_alpha: int = field(default=32, metadata={"help": "Alpha scaling LoRA (tipicamente 2x rank)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout LoRA."})


def render_minicpm(messages: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Testo identico a deploy_n100/minicpm-chat.jinja (senza BOS, lo aggiunge il tokenizer):
        <用户>{system}\n\n{user}<AI>{assistant}</s><用户>{user}<AI>{assistant}</s>
    Restituisce il testo e gli intervalli di caratteri (inizio, fine) di ogni
    risposta dell'assistente, </s> compreso: sono gli unici token con loss.
    """
    text = ""
    spans = []
    pending_system = ""
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "system":
            pending_system += content
        elif role == "user":
            text += "<用户>"
            if pending_system:
                text += pending_system + "\n\n"
                pending_system = ""
            text += content + "<AI>"
        elif role == "assistant":
            start = len(text)
            text += content + "</s>"
            spans.append((start, len(text)))
    return text, spans


class ChatMLDataset(Dataset):
    """
    Dataset supervisionato nel formato nativo di MiniCPM, anche multi-turno.
    La loss e' calcolata solo sui token delle risposte (e sui </s> finali).
    Gli esempi piu' lunghi di model_max_length vengono scartati: troncarli
    toglierebbe proprio la risposta, cioe' l'unica parte che addestra.
    """
    def __init__(self, data_path: str, tokenizer, model_max_length: int = 512):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Serve un tokenizer 'fast' (offset mapping) per mascherare le label sui caratteri.")

        self.examples = []
        dropped = 0
        for ex in raw:
            item = self.encode(ex.get("messages", []))
            if item is None:
                dropped += 1
            else:
                self.examples.append(item)
        print(f"    {data_path}: {len(self.examples)} esempi, {dropped} scartati (oltre {model_max_length} token o senza risposta)")

    def encode(self, messages: List[Dict]):
        text, spans = render_minicpm(messages)
        if not spans:
            return None
        enc = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        if len(input_ids) > self.model_max_length:
            return None
        labels = []
        for tok_id, (s, e) in zip(input_ids, enc["offset_mapping"]):
            inside = e > s and any(s >= a and e <= b for a, b in spans)
            labels.append(tok_id if inside else self.ignore_index)
        if all(l == self.ignore_index for l in labels):
            return None
        return input_ids, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        input_ids, labels = list(input_ids), list(labels)
        attention_mask = [1] * len(input_ids)

        # Padding a model_max_length
        pad_len = self.model_max_length - len(input_ids)
        eos_id = self.tokenizer.eos_token_id or 2
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        input_ids += [pad_token_id] * pad_len
        labels += [self.ignore_index] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def run_training():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"--> Caricamento Tokenizer da: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--> Configurazione BitsAndBytes per QLoRA a 4-bit (NF4, FP16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"--> Caricamento Modello Base: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # Preparazione per k-bit training e gradient checkpointing per risparmiare VRAM
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("--> Iniezione adattatori LoRA su tutti i moduli lineari...")
    # Tutti i moduli lineari (Attention + MLP) per massimizzare la capacità di ragionamento
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("--> Caricamento dei dataset...")
    train_dataset = ChatMLDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = None
    if data_args.eval_data_path and os.path.exists(data_args.eval_data_path):
        eval_dataset = ChatMLDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("--> Avvio del Fine-Tuning QLoRA su RTX 2080...")
    trainer.train()

    print(f"--> Salvataggio adattatori LoRA in: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("--> Training completato con successo!")


if __name__ == "__main__":
    run_training()
