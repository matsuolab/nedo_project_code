#!/bin/bash

#SBATCH --partition=g2 
#SBATCH --nodes=8
#SBATCH --gpus-per-node=0
#SBATCH --time=06:00:00
#SBATCH --nodelist=mlpre-g2-ghpc-5
#SBATCH --output=myjob.out
#SBATCH --error=error.out

./deduplicate output_cc100_ja result_cc100_ja/ 1000
