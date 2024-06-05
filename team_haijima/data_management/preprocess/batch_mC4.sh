#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=mC4_preprocess
#SBATCH --output=preprocess_mC4.out
#SBATCH --gpus-per-node=0
#SBATCH --mem=120GB
#SBATCH -c 47

# Command(s) goes here
# cd /persistentshare/storage/team_haijima/dataset_pre/20240410_CC100_filtering
source activate filter
pip install pybind11 cython

echo "install prepocess script"
cd preprocess
pip install .

cd ../
echo "start filtering"
cd ./preprocess/scripts
python preprocess_mC4.py --index_from=706 --index_to=1023 --num_worker=47
