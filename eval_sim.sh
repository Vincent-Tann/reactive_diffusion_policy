#!/bin/bash

# # DP for Square task
# python eval_sim_robot.py \
#       --config-name train_diffusion_unet_sim_image_workspace \
#       task=sim_square_image_dp_12fps \
#       +task.env_runner.output_dir=./output/videos \
#       +ckpt_path=/path/to/dp/checkpoint

# # RDP w. Force for Square task
# python eval_sim_robot.py \
#      --config-name train_latent_diffusion_unet_sim_image_workspace \
#      task=sim_square_image_wrench_ldp_24fps \
#      +task.env_runner.output_dir=./output/videos \
#      at=at_square \
#      +ckpt_path=/home/txs/Code/tactile/reactive_diffusion_policy/data/outputs/2025.04.16/23.11.42_train_latent_diffusion_unet_image_robosuite_square_image_wrench_ldp_0416231051/checkpoints/latest.ckpt \
#      at_load_dir=/home/txs/Code/tactile/reactive_diffusion_policy/data/outputs/2025.04.16/16.11.01_train_vae_robosuite_square_image_wrench_at_0416161058/checkpoints/latest.ckpt


python eval_sim_robot.py \
     --config-name train_latent_diffusion_unet_sim_image_workspace \
     task=robosuite_square_image_wrench_ldp \
     +task.env_runner.output_dir=./output/videos \
     at=at_square \
     +ckpt_path=/home/txs/Code/tactile/reactive_diffusion_policy/data/outputs/2025.04.16/23.11.42_train_latent_diffusion_unet_image_robosuite_square_image_wrench_ldp_0416231051/checkpoints/latest.ckpt \
     at_load_dir=/home/txs/Code/tactile/reactive_diffusion_policy/data/outputs/2025.04.16/16.11.01_train_vae_robosuite_square_image_wrench_at_0416161058/checkpoints/latest.ckpt