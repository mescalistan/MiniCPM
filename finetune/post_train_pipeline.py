# -*- coding: utf-8 -*-
import os, sys, time, subprocess, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "openbmb/MiniCPM-2B-dpo-fp16"
ADAPTER_PATH = "output/MiniCPM-2B-QLoRA-GitHub"
MERGED_DIR = "output/MiniCPM-2B-GitHub-Tied-FP16"
GGUF_F16 = "output/minicpm-2b-github-clean-f16.gguf"
GGUF_Q40 = "output/minicpm-2b-github-q4_0.gguf"

print("=" * 60)
print("POST-TRAIN PIPELINE: Merge -> Verify -> GGUF F16 -> Q4_0")
print("=" * 60)

# 1. Merge
print("\n[1/4] Caricamento Modello Base e Merge Adattatore...")
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

# Weight tying
merged.config.tie_word_embeddings = True
if hasattr(merged, "tie_weights"):
    merged.tie_weights()

print(f"Salvataggio merged model in {MERGED_DIR}...")
os.makedirs(MERGED_DIR, exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR)

# 2. PyTorch Test
print("\n[2/4] Verifica Inferenza PyTorch...")
test_prompt = "<用户>Come posso verificare i container attivi su Docker?<AI>"
inputs = tok(test_prompt, return_tensors="pt").to("cuda:0")
out = merged.generate(**inputs, max_new_tokens=45, do_sample=False)
gen_text = tok.decode(out[0], skip_special_tokens=False)
sys.stdout.buffer.write(b"PyTorch Output: " + gen_text.encode("utf-8") + b"\n")

del merged, base_model, peft_model
torch.cuda.empty_cache()

# 3. Conversione GGUF F16
print(f"\n[3/4] Conversione in GGUF FP16 ({GGUF_F16})...")
cmd_conv = [
    sys.executable,
    "llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR,
    "--outfile", GGUF_F16,
    "--outtype", "f16"
]
res = subprocess.run(cmd_conv, check=True)

# 4. Quantizzazione Q4_0
print(f"\n[4/4] Quantizzazione in Q4_0 ({GGUF_Q40})...")
cmd_quant = [
    os.path.abspath("llama_bin/llama-quantize.exe"),
    os.path.abspath(GGUF_F16),
    os.path.abspath(GGUF_Q40),
    "q4_0"
]
res = subprocess.run(cmd_quant, check=True)

# 5. Test locale con llama-cli
print("\n[5/5] Test di inferenza locale con llama-cli.exe...")
cmd_cli = [
    os.path.abspath("llama_bin/llama-cli.exe"),
    "-m", os.path.abspath(GGUF_Q40),
    "-p", "<用户>Come posso verificare i container attivi su Docker?<AI>",
    "-n", "45",
    "-ngl", "0",
    "--temp", "0.2"
]
subprocess.run(cmd_cli, check=True)

print("\n" + "=" * 60)
print("SUCCESS: Pipeline post-training completata!")
print(f"Modello pronto per il deploy: {GGUF_Q40}")
print("=" * 60)
