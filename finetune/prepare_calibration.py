# -*- coding: utf-8 -*-
"""
Testo di calibrazione per llama-imatrix nel formato reale del runtime
(<用户>...<AI>...</s>, vedi deploy_n100/minicpm-chat.jinja).
Il vecchio file usava ChatML (<|im_start|>), un formato che MiniCPM non vede mai.
post_train_multitask.py lo rigenera da solo; questo script serve per farlo a mano.
"""
import json
import sys

sys.path.insert(0, "finetune")
from train_qlora_win import render_minicpm  # noqa: E402

with open("finetune/data/train_multitask.json", "r", encoding="utf-8") as f:
    data = json.load(f)

full_calib = "\n".join(render_minicpm(ex["messages"])[0] for ex in data[:400])
with open("finetune/data/calibration_text.txt", "w", encoding="utf-8") as f:
    f.write(full_calib)

print(f"Calibration text written: {len(full_calib)} characters, ~{len(full_calib.split())} words")
