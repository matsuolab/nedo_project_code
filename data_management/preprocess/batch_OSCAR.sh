#!/bin/bash

# Command line options go here
#SBATCH --partition=g2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=OSCAR_preprocess
#SBATCH --output=preprocess_OSCAR.out
#SBATCH --gpus-per-node=0
#SBATCH --mem=120GB
#SBATCH -c 47

# Command(s) goes here
source activate filter
pip install pybind11 cython

echo "install prepocess script"
cd preprocess
pip install .

cd ../
echo "start filtering"
cd ./preprocess/scripts
python preprocess_OSCAR.py --index_from=95 --index_to=150 --num_worker=47
