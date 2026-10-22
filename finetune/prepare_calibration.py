# -*- coding: utf-8 -*-
import json

with open("finetune/data/train_custom.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sample_texts = []
for ex in data[:120]:
    block = []
    for msg in ex.get("messages", []):
        block.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    sample_texts.append("\n".join(block))

full_calib = "\n\n".join(sample_texts)
with open("finetune/data/calibration_text.txt", "w", encoding="utf-8") as f:
    f.write(full_calib)

print(f"Calibration text written: {len(full_calib)} characters, ~{len(full_calib.split())} words")
