# -*- coding: utf-8 -*-
"""
Post-training MiniCPM-2B Multi-Task: merge -> GGUF -> imatrix -> quantizzazione -> smoke test -> (deploy)

1. Merge dell'adattatore LoRA nel modello base FP16, rispettando la config del
   base: la variante "llama-format" ha lm_head separato (tie_word_embeddings
   False) e forzare il tying li' produce logit sbagliati, cioe' gibberish.
2. Conversione HF -> GGUF F16 (llama.cpp/convert_hf_to_gguf.py)
3. Importance matrix calcolata su testo nel formato reale del runtime
   (<用户>...<AI>...</s>), non ChatML come prima
4. Quantizzazione Q4_K_M (default) o IQ4_XS con --imatrix
5. Smoke test con llama-completion: fallisce se l'output contiene byte UTF-8
   rotti, cicli o marcatori di turno
6. Deploy su HOME-SERVER solo con --deploy (credenziali: chiave SSH o
   variabile HOME_SERVER_PASSWORD)

Uso:
  python finetune/post_train_multitask.py
  python finetune/post_train_multitask.py --quant IQ4_XS --ppl
  python finetune/post_train_multitask.py --skip-merge --deploy
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finetune"))

LLAMA_BIN = REPO_ROOT / "llama_bin"
if not LLAMA_BIN.exists():  # nel worktree i binari stanno nella checkout principale
    LLAMA_BIN = Path(os.environ.get("LLAMA_BIN", REPO_ROOT.parent / "llama_bin"))
CONVERTER = Path(os.environ.get("LLAMA_CPP_DIR", REPO_ROOT / "llama.cpp")) / "convert_hf_to_gguf.py"
PROMPTS = json.loads((REPO_ROOT / "home-server" / "ai-soc-app" / "lib" / "prompts.json").read_text(encoding="utf-8"))

REMOTE_HOST = os.environ.get("HOME_SERVER_HOST", "192.168.1.180")
REMOTE_USER = os.environ.get("HOME_SERVER_USER", "gabibbo")
REMOTE_MODEL_DIR = "/home/gabibbo/minicpm/models"
REMOTE_DEPLOY_DIR = "/home/gabibbo/minicpm/deploy_n100"


def tool(name: str) -> str:
    exe = LLAMA_BIN / (name + (".exe" if sys.platform == "win32" else ""))
    if not exe.exists():
        raise FileNotFoundError(f"{exe} non trovato (imposta LLAMA_BIN)")
    return str(exe)


def run(cmd, **kw):
    print("   $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------------------------------------------------------------------------
# 1. Merge
# ---------------------------------------------------------------------------
def merge(base: str, adapter: str, merged_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    tie = bool(getattr(cfg, "tie_word_embeddings", False))
    print(f"   base={base} tie_word_embeddings={tie}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.float16).merge_and_unload()
    model.eval()
    # Si conserva la scelta del modello base, senza forzarla
    model.config.tie_word_embeddings = tie
    if tie and hasattr(model, "tie_weights"):
        model.tie_weights()

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. Testo di calibrazione nel formato del runtime
# ---------------------------------------------------------------------------
def build_calibration(train_path: Path, out_path: Path, samples: int = 400) -> None:
    from train_qlora_win import render_minicpm

    data = json.loads(train_path.read_text(encoding="utf-8"))
    random.Random(7).shuffle(data)
    # Mix per task, cosi' l'imatrix vede chat, RAG, SOC, codice e documenti
    by_task = {}
    for ex in data:
        by_task.setdefault(ex.get("task", "chat"), []).append(ex)
    picked = []
    while len(picked) < samples and any(by_task.values()):
        for rows in by_task.values():
            if rows and len(picked) < samples:
                picked.append(rows.pop())
    text = "\n".join(render_minicpm(ex["messages"])[0] for ex in picked)
    out_path.write_text(text, encoding="utf-8")
    print(f"   calibrazione: {len(picked)} esempi, {len(text)} caratteri -> {out_path}")


# ---------------------------------------------------------------------------
# 5. Smoke test
# ---------------------------------------------------------------------------
SMOKE = [
    ("chat", "come stai ?"),
    ("chat", "Spiegami in due frasi cos'è un container Docker."),
    ("soc", "[TELEMETRIA]\nscore=92/100 (BUONO)\ncpu_temp=47C load=0.80/0.65/0.50\nram=3.4 GB/11.2 GB (30%)\n"
            "container=22/23 unhealthy=1 problemi=[mail-app:unhealthy]\nssh_falliti=2 ip_sospetti=0\n"
            "nginx_4xx=5 nginx_5xx=0\ncriticita=1 container in stato UNHEALTHY\n\n[DOMANDA]\nci sono container con problemi?"),
    ("rag", "[FONTI]\n[1] Kernel.org\nIl kernel Linux 6.12 integra lo scheduler sched_ext basato su eBPF e il supporto PREEMPT_RT.\n\n"
            "[2] Meteo\nDomani pioggia su Milano.\n\n[DOMANDA]\nQuali sono le novità di Linux 6.12?"),
]

DEGENERATE = [
    (re.compile(r"�"), "byte UTF-8 rotti"),
    (re.compile(r"(\w{2,6})\1{3,}"), "sillabe in ciclo"),
    (re.compile(r"<用户>|<AI>|<\|im_start\|>"), "marcatore di turno nell'output"),
    (re.compile(r"!\[[^\]]*\]\("), "immagine markdown inventata"),
]


def smoke_test(model_path: Path) -> bool:
    ok = True
    for kind, user in SMOKE:
        prompt = f"<用户>{PROMPTS[kind]}\n\n{user}<AI>"
        res = subprocess.run(
            [tool("llama-completion"), "-m", str(model_path), "-p", prompt, "-n", "120", "-no-cnv",
             "--temp", "0.3", "--top-p", "0.8", "--min-p", "0.05", "--repeat-penalty", "1.05",
             "--no-display-prompt", "-t", "4"],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=300,
        )
        out = res.stdout.decode("utf-8", errors="replace").strip()
        problems = [label for rx, label in DEGENERATE if rx.search(out)]
        status = "OK " if not problems else "KO "
        ok = ok and not problems
        print(f"\n   [{status}] {kind}: {user.splitlines()[-1]}\n   -> {out[:400]}")
        if problems:
            print(f"   !! {', '.join(problems)}")
    return ok


# ---------------------------------------------------------------------------
# 6. Deploy
# ---------------------------------------------------------------------------
def deploy(model_path: Path, remote_name: str) -> None:
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=os.environ.get("HOME_SERVER_PASSWORD"), look_for_keys=True, allow_agent=True)
    sftp = ssh.open_sftp()

    remote = f"{REMOTE_MODEL_DIR}/{remote_name}"
    size = model_path.stat().st_size
    t0 = time.time()

    def progress(done, total):
        mb = done / 2**20
        sys.stdout.write(f"\r   upload {mb:.0f}/{total / 2**20:.0f} MB ({mb / max(time.time() - t0, 1e-3):.1f} MB/s)  ")
        sys.stdout.flush()

    print(f"   upload {size / 2**20:.0f} MB -> {remote}")
    sftp.put(str(model_path), remote + ".tmp", callback=progress)
    print()

    def sh(cmd):
        _, out, err = ssh.exec_command(cmd)
        code = out.channel.recv_exit_status()
        text = out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")
        if code != 0:
            raise RuntimeError(f"{cmd}\n{text}")
        return text

    # Backup del modello con lo stesso nome, poi sostituzione atomica
    sh(f"[ -f {remote} ] && cp -p {remote} {remote}.prev || true")
    sh(f"mv {remote}.tmp {remote}")
    # Il compose legge il nome del file da MINICPM_MODEL in deploy_n100/.env
    sh(f"cd {REMOTE_DEPLOY_DIR} && (grep -v '^MINICPM_MODEL=' .env 2>/dev/null; echo 'MINICPM_MODEL={remote_name}') > .env.new && mv .env.new .env")
    # Il servizio si chiama llama-server (minicpm2b-n100 e' il nome del container)
    print(sh(f"cd {REMOTE_DEPLOY_DIR} && docker compose up -d --force-recreate llama-server && sleep 8 && docker ps --filter name=minicpm2b-n100"))
    sftp.close()
    ssh.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="openbmb/MiniCPM-2B-dpo-fp16")
    parser.add_argument("--adapter", default="output/MiniCPM-2B-QLoRA-Multitask")
    parser.add_argument("--merged-dir", default="output/MiniCPM-2B-Multitask-FP16")
    parser.add_argument("--train-data", default="finetune/data/train_multitask.json")
    parser.add_argument("--quant", default="Q4_K_M", choices=["Q4_K_M", "IQ4_XS", "Q5_K_M"])
    parser.add_argument("--imatrix-chunks", type=int, default=120)
    parser.add_argument("--skip-merge", action="store_true", help="usa il modello gia' unito in --merged-dir")
    parser.add_argument("--ppl", action="store_true", help="misura la perplexity F16 vs quantizzato sul testo di calibrazione")
    parser.add_argument("--deploy", action="store_true", help="carica il GGUF su HOME-SERVER e ricrea il container")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    f16 = out_dir / "minicpm-2b-multitask-f16.gguf"
    calib = out_dir / "calibration_minicpm.txt"
    imatrix = out_dir / "imatrix-multitask.gguf"
    final = out_dir / f"minicpm-2b-{args.quant.lower()}.gguf"

    print("[1/6] Merge LoRA -> FP16")
    if args.skip_merge:
        print("   saltato")
    else:
        merge(args.base, args.adapter, args.merged_dir)

    print("[2/6] HF -> GGUF F16")
    run([sys.executable, CONVERTER, args.merged_dir, "--outfile", f16, "--outtype", "f16"])

    print("[3/6] Importance matrix")
    build_calibration(REPO_ROOT / args.train_data, calib)
    run([tool("llama-imatrix"), "-m", f16, "-f", calib, "-o", imatrix, "-c", "512", "--chunks", args.imatrix_chunks, "-t", "8"])

    print(f"[4/6] Quantizzazione {args.quant}")
    run([tool("llama-quantize"), "--imatrix", imatrix, f16, final, args.quant])

    if args.ppl:
        for m in (f16, final):
            run([tool("llama-perplexity"), "-m", m, "-f", calib, "-c", "512", "--chunks", "40", "-t", "8"])

    print("[5/6] Smoke test")
    if not smoke_test(final):
        print("\nSmoke test FALLITO: il modello non viene distribuito.")
        sys.exit(2)

    print("[6/6] Deploy")
    if args.deploy:
        deploy(final, final.name)
    else:
        print(f"   saltato (usa --deploy). Modello pronto: {final}")


if __name__ == "__main__":
    main()
