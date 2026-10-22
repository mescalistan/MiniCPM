# Script PowerShell per avviare il fine-tuning QLoRA su RTX 2080 (Windows)
# Utilizza precisione FP16, 4-bit NF4, paged_adamw_8bit e target su tutti i moduli lineari.

$FormattedTime = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = "output/MiniCPM-2B-QLoRA-$FormattedTime"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Avvio Fine-Tuning QLoRA MiniCPM-2B su GPU RTX 2080" -ForegroundColor Cyan
Write-Host "Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

python finetune/train_qlora_win.py `
    --model_name_or_path "openbmb/MiniCPM-2B-dpo-bf16-llama-format" `
    --output_dir $OutputDir `
    --train_data_path "finetune/data/train_custom.json" `
    --eval_data_path "finetune/data/eval_custom.json" `
    --model_max_length 1024 `
    --lora_r 16 `
    --lora_alpha 32 `
    --lora_dropout 0.05 `
    --per_device_train_batch_size 2 `
    --per_device_eval_batch_size 2 `
    --gradient_accumulation_steps 8 `
    --learning_rate 1e-4 `
    --lr_scheduler_type "cosine" `
    --warmup_ratio 0.05 `
    --weight_decay 0.01 `
    --num_train_epochs 3 `
    --fp16 True `
    --optim "paged_adamw_8bit" `
    --logging_steps 10 `
    --save_strategy "steps" `
    --save_steps 200 `
    --evaluation_strategy "steps" `
    --eval_steps 200 `
    --save_total_limit 2 `
    --seed 42

Write-Host "`nTraining completato! Adattatore LoRA salvato in: $OutputDir" -ForegroundColor Green
Write-Host "Per effettuare il merge con il modello base, esegui:" -ForegroundColor Yellow
Write-Host "python finetune/merge_lora.py --base_model openbmb/MiniCPM-2B-dpo-bf16-llama-format --lora_adapter $OutputDir --output_dir output/MiniCPM-2B-Merged-FP16" -ForegroundColor White
