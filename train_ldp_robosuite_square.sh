#!/bin/bash

GPU_ID=0

TASK_NAME="peel"
DATASET_PATH="/home/txs/Code/tactile/reactive_diffusion_policy/robosuite_dataset/square_zarr"
LOGGING_MODE="online"

TIMESTAMP=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

# # find the latest checkpoint
# echo ""
# echo "Searching for the latest AT checkpoint..."
# TIMESTAMP="0411141624"
# AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints/latest.ckpt
# # AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -type d)/checkpoints/latest.ckpt

AT_LOAD_DIR="/home/txs/Code/tactile/reactive_diffusion_policy/data/outputs/2025.04.16/16.11.01_train_vae_robosuite_square_image_wrench_at_0416161058/checkpoints/latest.ckpt"

echo "AT_LOAD_DIR:"
echo "${AT_LOAD_DIR}"

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
    exit 1
fi

# Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=robosuite_square_image_wrench_ldp \
    task.dataset_path=${DATASET_PATH} \
    task.name=robosuite_square_image_wrench_ldp_${TIMESTAMP} \
    at=at_square \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE}