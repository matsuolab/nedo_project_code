#!/bin/bash

#SBATCH --time=30-00:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-11
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=upload
#SBATCH --output=upload3.out

master_port=$((10000 + (${SLURM_JOB_ID} % 50000)))

cd ~
deepspeed --no_local_rank --master_port ${master_port} pretrain/utils/ckpt_upload.py \
--ckpt-path /storage4/work/kohmoto/dpo_merge_20240526_final_ckpt500 \
--repo-name JINIAC/JINIAC-5B-sft_configuration-3_prod-checkpoint-100-dpo_merge_20240526_final_ckpt500