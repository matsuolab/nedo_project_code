#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=cc100_preprocess
#SBATCH --output=preprocess_cc100.out
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
python preprocess_cc100.py --index_from=1 --index_to=15 --num_worker=47
