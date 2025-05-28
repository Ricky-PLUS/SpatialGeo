#!/bin/bash

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /root/private_data/MyCode/spatialLLaVA/llavamodel/llava-v1.5-7b \
    --version v1 \
    --data_path /root/private_data/MyCode/dataset/llava_v1_5_mix665k.json \
    --image_folder /root/private_data/MyCode/dataset \
    --vision_tower /root/private_data/MyCode/spatialLLaVA/llavamodel/clip-vit-large-patch14-336 \
    --pretrain_moge_mm_mlp_adapter /root/private_data/MyCode/spatialLLaVA/checkpointsmoge/llava-v1.5-7b-moge_projector/moge_mm_projector.bin \
    --freeze_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpointsmoge/llava-v1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb



# finetune
# --learning_rate 2e-5
# --output_dir ./checkpoints/llava-v1.5-13b
# ***--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 

# finetune_task_lora
# --output_dir ./checkpoints/llava-v1.5-13b-task-lora
# ***--pretrain_moge_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin 
# --model_name_or_path liuhaotian/llava-v1.5-13b

# pretrainmoge
# ***--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5
# ***--pretrain_moge_mm_mlp_adapter /root/private_data/MyCode/spatialLLaVA/checkpoints/llava-v1.6-7b-moge_projector/moge_mm_projector.bin 
# --tune_mm_mlp_adapter True
# ***--image_aspect_ratio pad
# ***--group_by_modality_length True
# --per_device_train_batch_size 32
# --save_steps 1000
# --learning_rate 1e-3 