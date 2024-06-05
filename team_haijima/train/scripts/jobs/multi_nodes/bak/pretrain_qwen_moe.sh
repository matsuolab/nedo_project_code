#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[6-8]
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=pretrain_qwen_moe_14b
#SBATCH --mem=1000G
#SBATCH --output=/storage3/jobs_outs/%x_%j.out

# Command(s) goes here
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .cuda12
/storage3/GENIAC_haijima/train/scripts/step2_pretrain_model/gcp_node-2_gpu/pretrain_qwen_moe.sh --save_interval 100
