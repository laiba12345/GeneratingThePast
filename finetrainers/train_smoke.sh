#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export HF_HUB_DISABLE_XET=1
export HF_HOME=/Data/Laiba-Maddy-Project/Project2/hf_cache

torchrun --nproc_per_node=1 train.py \
  --parallel_backend ptd \
  --pp_degree 1 \
  --dp_degree 1 \
  --dp_shards 1 \
  --cp_degree 1 \
  --tp_degree 1 \
  --model_name ltx_video \
  --pretrained_model_name_or_path /Data/Laiba-Maddy-Project/Project2/models/LTX-Video\
  --dataset_config /Data/Laiba-Maddy-Project/Project2/Dataset/dataset_config.json \
  --flow_weighting_scheme logit_normal \
  --enable_precomputation \
  --precomputation_items 32 \
  --training_type lora \
  --seed 42 \
  --batch_size 1 \
  --train_steps 5250 \
  --rank 32 \
  --lora_alpha 32 \
  --target_modules to_q to_k to_v to_out.0 \
  temporal_transformer_blocks.0.attn1.to_q \
  temporal_transformer_blocks.0.attn1.to_k \
  temporal_transformer_blocks.0.attn1.to_v \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --checkpointing_steps 250 \
  --checkpointing_limit 4 \
  --enable_slicing \
  --optimizer adamw \
  --lr 3e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --beta1 0.9 \
  --beta2 0.99 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --tracker_name ltx-reverse-smoke \
  --output_dir /Data/Laiba-Maddy-Project/Project2/outputs/output5 \
  --report_to wandb