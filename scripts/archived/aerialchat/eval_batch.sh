#!/bin/bash

CHUNKS=2
EXP_NAME=debug-llavaov-env-new-2-zero3/epoch_4
MODEL_DIR=./output/aerialchat/${EXP_NAME}
for IDX in {0..1}; do
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.aerialchat.inference \
        --model-base ./models/llava-onevision-qwen2-7b-ov \
        --model-path-stage1 ./checkpoints/llava-onevision-qwen2-7b-ov-max1-lora-geochat-stage2 \
        --model-path-stage2 ${MODEL_DIR} \
        --eval-data-path ./data/train_1094.json \
        --image-folder ./data/train_images \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done
