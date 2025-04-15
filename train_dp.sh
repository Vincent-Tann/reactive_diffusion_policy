#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_peel_image_gelsight_emb_absolute_12fps \
    task.dataset_path=/home/txs/Code/tactile/reactive_diffusion_policy/reactive_diffusion_policy_dataset/dataset_mini/peel_v3_downsample1_zarr \
    task.name=real_peel_image_gelsight_emb_absolute_12fps \
    logging.mode=online