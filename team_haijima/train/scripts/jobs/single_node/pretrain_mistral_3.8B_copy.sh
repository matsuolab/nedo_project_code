#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[7]
#SBATCH --job-name=pretrain_mistral_3.8B
#SBATCH --mem=1500G
#SBATCH --output=/storage3/jobs_outs/%x_%j.out

# Command(s) goes here
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .cuda12
/storage3/GENIAC_haijima/train/scripts/step2_pretrain_model/gcp_node-1_gpu/pretrain_mistral_3.8b_copy.sh --save_interval 100
