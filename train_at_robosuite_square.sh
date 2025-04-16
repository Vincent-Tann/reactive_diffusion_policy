#!/bin/bash

GPU_ID=0

TASK_NAME="square"
DATASET_PATH="/home/txs/Code/tactile/reactive_diffusion_policy/robosuite_dataset/square_zarr"
LOGGING_MODE="online"

TIMESTAMP=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=robosuite_square_image_wrench_at \
    task.dataset_path=${DATASET_PATH} \
    task.name=robosuite_square_image_wrench_at_${TIMESTAMP} \
    at=at_square \
    logging.mode=${LOGGING_MODE}