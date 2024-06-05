#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[7-8]
#SBATCH --job-name=test_run
#SBATCH --mem=1500G
#SBATCH --output=/storage3/jobs_outs/%x-%j.out

# Command(s) goes here
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .cuda12
/storage3/GENIAC_haijima/train/scripts/step2_pretrain_model/gcp_node-1_gpu/test_run2.sh --save_interval 700

