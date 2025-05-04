#!/bin/bash

CHUNKS=2
EXP_NAME=debug-llavaov-env-new-2-zero3/epoch_4
MODEL_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/output/aerialchat/${EXP_NAME}
for IDX in {0..1}; do
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.aerialchat.inference \
        --model-base /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/models/lmms-lab/llava-onevision-qwen2-7b-ov \
        --model-path-stage1 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/output/geochat/llava-onevision-qwen2-7b-ov-max1-lora-09-09-geochat-stage2 \
        --model-path-stage2 ${MODEL_DIR} \
        --eval-data-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/huitianrui/projects/project_2024/code/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_1094.json \
        --image-folder /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/huitianrui/projects/project_2024/code/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done
