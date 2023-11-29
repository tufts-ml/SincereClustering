#!/bin/bash
#SBATCH -p ccgpu,gpu
#SBATCH -t 0-3:0:0
#SBATCH --gres=gpu:a100:1
#SBATCH -c 9

set -e
set -x

CUDA_VISIBLE_DEVICES=0 pipenv run python train.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --exp_name cub_simgcd
