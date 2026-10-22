# -*- coding: utf-8 -*-
"""
Web UI moderno per MiniCPM-2B su GPU RTX 2080 (Windows).
Interfaccia basata su Gradio con supporto streaming e chat interattiva.
"""

import sys
import argparse
from threading import Thread
import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def extract_text(content) -> str:
    """Estrae testo puro sia da stringhe che da blocchi multimodali/liste di Gradio 6."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", str(item)))))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", str(content))))
    return str(content)


def create_ui(model_path: str = "openbmb/MiniCPM-2B-dpo-fp16", port: int = 7860):
    print("=" * 60)
    print(f"Caricamento MiniCPM-2B su GPU (RTX 2080): {model_path}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    def user_submit(user_msg, chat_hist):
        text = extract_text(user_msg).strip()
        if not text:
            return "", chat_hist
        if chat_hist is None:
            chat_hist = []
        chat_hist = chat_hist + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": ""}
        ]
        return "", chat_hist

    def bot_stream(chat_hist, sys_p, temp, tp, max_t):
        if not chat_hist or len(chat_hist) < 2:
            return

        messages = []
        sys_text = extract_text(sys_p).strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        # Prepara lo storico con contenuto garantito come stringa pura
        for msg in chat_hist[:-1]:
            role = msg.get("role", "user")
            content = extract_text(msg.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content})

        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_t),
            temperature=float(temp),
            top_p=float(tp),
            repetition_penalty=1.05,
            do_sample=True if float(temp) > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            chat_hist[-1]["content"] = accumulated
            yield chat_hist

    with gr.Blocks(title="MiniCPM-2B Local Web UI") as demo:
        gr.Markdown(
            """
            # 🚀 MiniCPM-2B (Local Inference su RTX 2080)
            Modello eseguito in locale in precisione **FP16** con accelerazione hardware CUDA.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=520, label="Conversazione")
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Scrivi qui la tua domanda (es. 'Spiegami la relatività generale in 3 punti')...",
                        label="Messaggio",
                        lines=2,
                        scale=8
                    )
                    send_btn = gr.Button("Invia 📤", variant="primary", scale=2)
                
                with gr.Row():
                    clear_btn = gr.Button("Pulisci Chat 🧹")

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parametri Generazione")
                system_prompt = gr.Textbox(
                    value="Sei un assistente AI cordiale, diretto, altamente logico e preciso. Rispondi in italiano.",
                    label="System Prompt",
                    lines=3
                )
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-P")
                max_tokens = gr.Slider(128, 2048, value=1024, step=64, label="Max New Tokens")

        msg_input.submit(
            user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_stream, [chatbot, system_prompt, temperature, top_p, max_tokens], chatbot
        )

        send_btn.click(
            user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_stream, [chatbot, system_prompt, temperature, top_p, max_tokens], chatbot
        )

        clear_btn.click(lambda: [], None, chatbot, queue=False)

    print(f"\n--> Web UI pronta! Avvio su http://127.0.0.1:{port}\n")
    demo.queue().launch(server_name="127.0.0.1", server_port=port, inbrowser=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-2B-dpo-fp16")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_ui(model_path=args.model_path, port=args.port)
