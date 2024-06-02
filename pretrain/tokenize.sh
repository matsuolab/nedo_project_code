#!/bin/bash

#SBATCH --time=30-00:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-10
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1800GB
#SBATCH --job-name=tokenize
#SBATCH --output=tok3.out

master_port=$((10000 + (${SLURM_JOB_ID} % 50000)))

cd ~/pretrain
deepspeed --no_local_rank --master_port ${master_port}  train/run_clm.py train/train_config.json --deepspeed --deepspeed_config config/ds_config.json