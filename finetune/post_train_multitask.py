# -*- coding: utf-8 -*-
"""
Post-Training Automation for MiniCPM-2B Multi-Task:
1. Merge LoRA adapter into FP16 base model with weight-tying preservation
2. Convert Hugging Face model to GGUF FP16
3. Quantize GGUF to Q4_K_M (or Q4_0 with imatrix)
4. Local test inference with llama-cli.exe
5. Upload via SFTP to Intel N100 HOME-SERVER (192.168.1.180)
6. Restart minicpm2b-n100 container
"""

import os
import sys
import subprocess
import time
import torch
import paramiko
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "openbmb/MiniCPM-2B-dpo-fp16"
ADAPTER_PATH = "output/MiniCPM-2B-QLoRA-Multitask"
MERGED_DIR = "output/MiniCPM-2B-Multitask-FP16"
GGUF_F16 = "output/minicpm-2b-multitask-f16.gguf"
GGUF_FINAL = "output/minicpm-2b-q4_k_m.gguf"
REMOTE_HOST = "192.168.1.180"
REMOTE_USER = "gabibbo"
REMOTE_PASS = "225569"
REMOTE_MODEL_PATH = "/home/gabibbo/minicpm/models/minicpm-2b-q4_k_m.gguf"

def main():
    print("=" * 65)
    print("POST-TRAIN MULTI-TASK PIPELINE: MERGE -> GGUF -> DEPLOY N100")
    print("=" * 65)

    # 1. Merge LoRA weights into FP16 base model
    print("\n[1/5] Caricamento Modello Base e Merge Adattatori LoRA...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    merged = peft_model.merge_and_unload()
    merged.eval()

    # Weight tying preservation
    merged.config.tie_word_embeddings = True
    if hasattr(merged, "tie_weights"):
        merged.tie_weights()

    print(f"Salvataggio merged model in {MERGED_DIR}...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tok.save_pretrained(MERGED_DIR)

    del merged, base_model, peft_model
    torch.cuda.empty_cache()

    # 2. Conversione Hugging Face -> GGUF FP16
    print(f"\n[2/5] Conversione in GGUF FP16 ({GGUF_F16})...")
    cmd_conv = [
        sys.executable,
        "llama.cpp/convert_hf_to_gguf.py",
        MERGED_DIR,
        "--outfile", GGUF_F16,
        "--outtype", "f16"
    ]
    subprocess.run(cmd_conv, check=True)

    # 3. Quantizzazione in Q4_K_M
    print(f"\n[3/5] Quantizzazione in Q4_K_M ({GGUF_FINAL})...")
    cmd_quant = [
        os.path.abspath("llama_bin/llama-quantize.exe"),
        os.path.abspath(GGUF_F16),
        os.path.abspath(GGUF_FINAL),
        "q4_k_m"
    ]
    subprocess.run(cmd_quant, check=True)

    # 4. Test rapido con llama-cli
    print("\n[4/5] Test di inferenza con llama-cli.exe...")
    cmd_cli = [
        os.path.abspath("llama_bin/llama-cli.exe"),
        "-m", os.path.abspath(GGUF_FINAL),
        "-p", "<用户>Come posso verificare i container attivi su Docker?<AI>",
        "-n", "45",
        "-ngl", "0",
        "--temp", "0.2"
    ]
    subprocess.run(cmd_cli, check=True, stdin=subprocess.DEVNULL)

    # 5. Deploy via SFTP su Intel N100 HOME-SERVER
    print(f"\n[5/5] Deploy del modello quantizzato su HOME-SERVER ({REMOTE_HOST})...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASS)
    sftp = ssh.open_sftp()

    file_size = os.path.getsize(GGUF_FINAL)
    print(f"Dimensione file da trasferire: {file_size / (1024*1024):.2f} MB")

    t0 = time.time()
    def progress_callback(transferred, total):
        pct = (transferred / total) * 100
        mb_trans = transferred / (1024 * 1024)
        mb_tot = total / (1024 * 1024)
        elapsed = time.time() - t0
        speed = mb_trans / elapsed if elapsed > 0 else 0
        sys.stdout.write(f"\rTrasferimento: {mb_trans:.1f}/{mb_tot:.1f} MB ({pct:.1f}%) a {speed:.1f} MB/s   ")
        sys.stdout.flush()

    sftp.put(GGUF_FINAL, REMOTE_MODEL_PATH + ".tmp", callback=progress_callback)
    print("\nUpload completato. Spostamento atomico del file...")
    # Atomic rename on server
    ssh.exec_command(f"mv {REMOTE_MODEL_PATH}.tmp {REMOTE_MODEL_PATH}")

    sftp.close()

    print("Riavvio del container minicpm2b-n100 sul server...")
    stdin, stdout, stderr = ssh.exec_command("cd /home/gabibbo/minicpm/deploy_n100 && docker compose restart minicpm2b-n100 && sleep 5 && docker ps --filter name=minicpm2b-n100")
    print(stdout.read().decode('utf-8'))
    ssh.close()

    print("\n" + "=" * 65)
    print("DEPLOY MULTI-TASK SU HOME-SERVER COMPLETATO CON SUCCESSO!")
    print("=" * 65)

if __name__ == "__main__":
    main()
