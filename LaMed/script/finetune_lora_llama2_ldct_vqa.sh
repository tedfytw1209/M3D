#!/bin/bash

# run "accelerate config" first!

accelerate launch LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path /orange/chenaokun1990/tienyu/huggingface/hub/models--GoodBaiBai88--M3D-LaMed-Llama-2-7B/snapshots/87f6f56001c2dd4d005c0d10c17edba4ffbf36a5/ \
    --vqa_yn_data_train_path /blue/chenaokun1990/tienyuchang/M3D/Data/data/VQA_large_nodule_train_nii_wunk.jsonl \
    --model_type llama2 \
    --lora_enable True \
    --vision_tower vit3d \
    --only-vqa \
    --bf16 True \
    --output_dir /red/chenaokun1990/tienyu/m3d_models/m3d_vqa_lora_2ep \
    --model_max_length 512 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to wandb