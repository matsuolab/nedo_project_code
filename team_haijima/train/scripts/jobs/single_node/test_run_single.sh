#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=slurm0-a3-ghpc-6
#SBATCH --job-name=test_run
#SBATCH --mem=300G

# Command(s) goes here
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .venv
/storage3/GENIAC_haijima/train/scripts/step2_pretrain_model/gcp_node-1_gpu/test_run2.sh --save_interval 700

