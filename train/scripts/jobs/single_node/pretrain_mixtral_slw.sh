#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[6-8]
#SBATCH --job-name=pretrain_mixtral_7bx8_slw
#SBATCH --mem=1500G
#SBATCH --output=/storage3/jobs_outs/%x_%j.out

# Command(s) goes here
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .venv
/storage3/GENIAC_haijima/train/scripts/step2_pretrain_model/gcp_node-1_gpu/pretrain_mixtral_slw.sh --save_interval 700
