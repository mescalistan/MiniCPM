# -*- coding: utf-8 -*-
"""
Interactive CLI Chat per MiniCPM-2B su GPU RTX 2080 (Windows).
Supporta streaming in tempo reale dei token e memoria multi-turno.
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from threading import Thread

# Forza UTF-8 sulla console Windows per visualizzare correttamente tutti i caratteri
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def interactive_chat(model_id: str = "openbmb/MiniCPM-2B-dpo-fp16", max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.8):
    print("=" * 65)
    print(f" Inizializzazione MiniCPM-2B su GPU (RTX 2080 - FP16)")
    print(f" Modello: {model_id}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Dispositivo: {device.upper()}")
    if device == "cuda":
        print(f"--> GPU: {torch.cuda.get_device_name(0)}")

    print("--> Caricamento del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("--> Caricamento del modello in FP16 su GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    print("\n" + "=" * 65)
    print(" Modello pronto! Comandi speciali disponibili:")
    print("   /clear   -> Azzera la memoria della conversazione")
    print("   /exit    -> Termina la sessione")
    print("=" * 65 + "\n")

    history = []

    while True:
        try:
            user_input = input("\n\033[1;36mTu > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChiusura sessione.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
            print("Arrivederci!")
            break

        if user_input.lower() == "/clear":
            history = []
            print("\033[1;33m[Memoria della conversazione azzerata]\033[0m")
            continue

        # Usa il metodo nativo model.chat ottimizzato per MiniCPM con streaming
        # Prepariamo la richiesta
        current_history = list(history)
        current_history.append({"role": "user", "content": user_input})
        prompt_str = tokenizer.apply_chat_template(current_history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("\033[1;32mMiniCPM > \033[0m", end="", flush=True)
        response_text = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            response_text += chunk
        print()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text.strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM-2B Local CLI Chat")
    parser.add_argument(
        "--model_path",
        type=str,
        default="openbmb/MiniCPM-2B-dpo-fp16",
        help="ID Hugging Face o percorso del modello"
    )
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)

    args = parser.parse_args()
    interactive_chat(
        model_id=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
