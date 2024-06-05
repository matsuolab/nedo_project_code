#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[6-8]
#SBATCH --job-name=pretrain_mistral_9b
#SBATCH --mem=1500G
#SBATCH --output=/storage3/jobs_outs/%x_%j.out

# Command(s) goes here
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .ds0_14_2
/storage3/GENIAC_haijima/train/scripts/step2_pretrain_model/gcp_node-2_gpu/pretrain_mistral_9b_slw.sh --save_interval 600
