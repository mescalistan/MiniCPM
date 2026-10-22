# Script PowerShell per avviare il fine-tuning QLoRA Multi-Task su RTX 2080 (Windows)
# Configurazione VRAM Ultra-Lean per esecuzione 100% in VRAM fisica senza paging WDDM.
#
# Prima: python finetune/prepare_multitask_dataset.py --total 4000
# Dopo:  python finetune/post_train_multitask.py
#
# model_max_length 1024: gli esempi RAG con 3 fonti superano i 384 token del
# vecchio valore e venivano troncati proprio sulla risposta. Con batch 1,
# gradient checkpointing e NF4 il 2B resta sotto ~6.5 GB di VRAM.
# 1 epoca su ~4000 esempi unici (~950 step) al posto di 150 step fissi.

param(
    [string]$BaseModel = "openbmb/MiniCPM-2B-dpo-fp16",
    [string]$OutputDir = "output/MiniCPM-2B-QLoRA-Multitask"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Avvio Fine-Tuning QLoRA MiniCPM-2B Multi-Task su RTX 2080" -ForegroundColor Cyan
Write-Host "Base:       $BaseModel" -ForegroundColor Cyan
Write-Host "Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

.\.venv\Scripts\python.exe finetune/train_qlora_win.py `
    --model_name_or_path $BaseModel `
    --output_dir $OutputDir `
    --train_data_path "finetune/data/train_multitask.json" `
    --eval_data_path "finetune/data/eval_multitask.json" `
    --model_max_length 1024 `
    --lora_r 16 `
    --lora_alpha 32 `
    --lora_dropout 0.05 `
    --per_device_train_batch_size 1 `
    --per_device_eval_batch_size 1 `
    --gradient_accumulation_steps 4 `
    --learning_rate 1e-4 `
    --lr_scheduler_type "cosine" `
    --warmup_ratio 0.03 `
    --weight_decay 0.01 `
    --num_train_epochs 1 `
    --fp16 True `
    --optim "paged_adamw_8bit" `
    --logging_steps 20 `
    --eval_strategy "steps" `
    --eval_steps 200 `
    --save_strategy "steps" `
    --save_steps 200 `
    --save_total_limit 2 `
    --load_best_model_at_end True `
    --metric_for_best_model "eval_loss" `
    --report_to "none" `
    --seed 42

if ($LASTEXITCODE -ne 0) { Write-Host "Training fallito." -ForegroundColor Red; exit $LASTEXITCODE }
Write-Host "`nTraining completato! Adattatore LoRA salvato in: $OutputDir" -ForegroundColor Green
