#!/bin/bash

#SBATCH --time=30-00:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-10
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --job-name=JINIAC
#SBATCH --output=jiniac_moe_ja_2.out

master_port=$((10000 + (${SLURM_JOB_ID} % 50000)))

cd ~/pretrain
deepspeed --no_local_rank --master_port ${master_port} train/run_clm.py train/train_config.json --deepspeed --deepspeed_config config/ds_config.json