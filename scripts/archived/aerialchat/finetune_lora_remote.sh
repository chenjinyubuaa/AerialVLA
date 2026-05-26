#!/bin/bash

PREFIX=aerialchat
EXP_NAME=llavaov-lora-aerialchat-dagger-old-with-eval-grid-v2

PROMPT_VERSION="qwen_1_5"

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

OUTPUT_DIR=./output/${PREFIX}/${EXP_NAME}

deepspeed llava/train/aerialchat/train.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./models/llava-onevision-qwen2-7b-ov \
    --version ${PROMPT_VERSION} \
    --data_path ./data/train_aerial_instrucion_grid_9_altitude_20.json \
    --image_folder ./data/train_images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_1 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 7000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 8 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb \
    --training_stage 2 \
    --use_dagger True \
    --dagger_epoch_num 20 \
    --dagger_data_rate 2. \
    --use_grid_action True \
    --grid_offset_size 9 \
    --grid_altitude_size 20 \
    --stage1_path ./checkpoints/llava-onevision-qwen2-7b-ov-max1-lora-geochat-stage2 \
    --evaluate_in_training True \
    --eval_data_path ./data/val_unseen_data.json
