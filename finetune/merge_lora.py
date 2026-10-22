# -*- coding: utf-8 -*-
"""
Script per il merge degli adattatori LoRA nel modello base MiniCPM-2B (FP16).
Genera una cartella modello standalone Hugging Face pronta per la conversione in GGUF.
Preserva rigorosamente il tying di lm_head.weight <-> model.embed_tokens.weight.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_export(base_model_path: str, lora_adapter_path: str, output_path: str, device: str = "cpu"):
    print("=" * 60)
    print("MiniCPM-2B: Merge LoRA Adapter -> Modello Base Standalone (FP16)")
    print("=" * 60)
    print(f"Modello Base:    {base_model_path}")
    print(f"Adattatore LoRA: {lora_adapter_path}")
    print(f"Directory Output:{output_path}")
    print(f"Device per Merge:{device}")

    print("\n1. Caricamento Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("\n2. Caricamento Modello Base in FP16 (senza quantizzazione)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    if hasattr(base_model, "tie_weights"):
        base_model.tie_weights()

    print("\n3. Caricamento dell'Adattatore LoRA...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.float16,
        device_map=device
    )

    print("\n4. Esecuzione del merge dei pesi (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()
    
    # Rigoroso weight-tying per MiniCPM-2B LlamaForCausalLM
    print("--> Sincronizzazione esplicita di lm_head.weight con model.embed_tokens.weight...")
    merged_model.config.tie_word_embeddings = True
    merged_model.lm_head.weight.data.copy_(merged_model.model.embed_tokens.weight.data)
    merged_model.tie_weights()

    merged_model.eval()

    print(f"\n5. Salvataggio del modello esportato in: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("SUCCESS: Modello standalone FP16 salvato con successo!")
    print(f"Pronto per convert_hf_to_gguf.py in: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="openbmb/MiniCPM-2B-dpo-fp16",
        help="Path o HuggingFace ID del modello base (default: openbmb/MiniCPM-2B-dpo-fp16)"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="Percorso della directory contenente l'adattatore LoRA addestrato (output di train_qlora_win.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/MiniCPM-2B-HighDensity-FP16",
        help="Percorso dove salvare il modello standalone FP16"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device su cui eseguire il merge (CPU è raccomandato per evitare limiti di VRAM)"
    )

    args = parser.parse_args()
    merge_and_export(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output_dir,
        device=args.device
    )
