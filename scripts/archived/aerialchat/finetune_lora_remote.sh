#!/bin/bash

PREFIX=aerialchat
EXP_NAME=llavaov-lora-aerialchat-dagger-old-with-eval-grid-v2

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################


PROMPT_VERSION="qwen_1_5"

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

###### for remote env setting ######
# ln -s /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/conda /home/sankuai/conda
# pip install -e .
###### for remote env setting ######

OUTPUT_DIR=${ROOT}/output/${PREFIX}/${EXP_NAME}

export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1

deepspeed llava/train/aerialchat/train.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/models/lmms-lab/llava-onevision-qwen2-7b-ov \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/train_aerial_instrucion_grid_9_altitude_20.json \
    --image_folder /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/huitianrui/projects/project_2024/code/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images \
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
    --stage1_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/output/geochat/llava-onevision-qwen2-7b-ov-max1-lora-09-09-geochat-stage2 \
    --evaluate_in_training True \
    --eval_data_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/huitianrui/projects/project_2024/code/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_unseen_data.json