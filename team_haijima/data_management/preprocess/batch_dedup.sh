#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=dedup
#SBATCH --output=dedup.out
#SBATCH --gpus-per-node=0
#SBATCH --mem=240GB
#SBATCH -c 1

# Command(s) goes here
python dedup.py

