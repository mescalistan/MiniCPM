# -*- coding: utf-8 -*-
"""
Script ottimizzato per Fine-Tuning QLoRA di MiniCPM-2B su Windows con GPU NVIDIA (RTX 2080 8GB).
Caratteristiche:
- Quantizzazione 4-bit NF4 con BitsAndBytes (VRAM < 5 GB)
- Targeting di tutti i moduli lineari (q, k, v, o, gate, up, down)
- Precisione di calcolo FP16 ottimale per Tensor Core Turing (sm_75)
- Supporto dataset formato ChatML con mascheramento prompt utente
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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


class ChatMLDataset(Dataset):
    """
    Dataset supervisionato ottimizzato per la sintassi nativa di MiniCPM:
    <用户>[system_prompt\n\n]user_query<AI>assistant_response</s>
    Calcola la loss unicamente sui token dell'assistente e sul token finale </s> (-100 sul resto).
    """
    def __init__(self, data_path: str, tokenizer, model_max_length: int = 512):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        messages = example.get("messages", [])

        system_content = ""
        user_content = ""
        assistant_content = ""

        for m in messages:
            role = m.get("role", "")
            if role == "system":
                system_content = m.get("content", "").strip()
            elif role == "user":
                user_content = m.get("content", "").strip()
            elif role == "assistant":
                assistant_content = m.get("content", "").strip()

        # Costruzione del prompt nativo MiniCPM
        if system_content:
            prompt_text = f"<用户>{system_content}\n\n{user_content}<AI>"
        else:
            prompt_text = f"<用户>{user_content}<AI>"

        resp_text = f"{assistant_content}</s>"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        resp_ids = self.tokenizer.encode(resp_text, add_special_tokens=False)

        # Se resp_ids non termina con eos_id, lo aggiungiamo
        eos_id = self.tokenizer.eos_token_id or 2
        if len(resp_ids) == 0 or resp_ids[-1] != eos_id:
            resp_ids.append(eos_id)

        input_ids = prompt_ids + resp_ids
        labels = [self.ignore_index] * len(prompt_ids) + resp_ids

        # Troncamento alla lunghezza massima
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        attention_mask = [1] * len(input_ids)

        # Padding a model_max_length
        pad_len = self.model_max_length - len(input_ids)
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
