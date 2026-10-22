# Script PowerShell per avviare il fine-tuning QLoRA Multi-Task su RTX 2080 (Windows)
# Configurazione VRAM Ultra-Lean per esecuzione 100% in VRAM fisica senza paging WDDM.

$OutputDir = "output/MiniCPM-2B-QLoRA-Multitask"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Avvio Fine-Tuning QLoRA MiniCPM-2B Multi-Task su RTX 2080" -ForegroundColor Cyan
Write-Host "Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

.\.venv\Scripts\python.exe finetune/train_qlora_win.py `
    --model_name_or_path "openbmb/MiniCPM-2B-dpo-fp16" `
    --output_dir $OutputDir `
    --train_data_path "finetune/data/train_multitask.json" `
    --eval_data_path "finetune/data/eval_multitask.json" `
    --model_max_length 384 `
    --lora_r 16 `
    --lora_alpha 32 `
    --lora_dropout 0.05 `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 4 `
    --learning_rate 2e-4 `
    --lr_scheduler_type "cosine" `
    --warmup_ratio 0.05 `
    --weight_decay 0.01 `
    --max_steps 150 `
    --fp16 True `
    --optim "paged_adamw_8bit" `
    --logging_steps 10 `
    --save_strategy "no" `
    --evaluation_strategy "no" `
    --seed 42

Write-Host "`nTraining completato! Adattatore LoRA salvato in: $OutputDir" -ForegroundColor Green
