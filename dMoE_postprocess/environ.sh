#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=slurm0-a3-ghpc-20
#SBATCH --job-name=env
#SBATCH --output=%x_%j.out
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=48

. ~/miniconda3/etc/profile.d/conda.sh

conda create --name postprocess python=3.9 -y

mkdir -p ~/miniconda3/envs/postprocess/etc/conda/activate.d
echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > ~/miniconda3/envs/postprocess/etc/conda/activate.d/edit_environment_variable.sh
echo 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/postprocess/lib:$LD_LIBRARY_PATH"' >> ~/miniconda3/envs/postprocess/etc/conda/activate.d/edit_environment_variable.sh
chmod +x ~/miniconda3/envs/postprocess/etc/conda/activate.d/edit_environment_variable.sh

# Python仮想環境を無効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を元に戻すように設定。
mkdir -p ~/miniconda3/envs/postprocess/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > ~/miniconda3/envs/postprocess/etc/conda/deactivate.d/rollback_environment_variable.sh
echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> ~/miniconda3/envs/postprocess/etc/conda/deactivate.d/rollback_environment_variable.sh
chmod +x ~/miniconda3/envs/postprocess/etc/conda/deactivate.d/rollback_environment_variable.sh

conda activate postprocess

conda install pytorch==2.2.0 -y
pip install "transformers>=4.34.0" sentencepiece "tokenizers>=0.14.0" "protobuf<3.21.0" 
