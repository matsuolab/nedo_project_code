#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --time=30-00:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=slurm0-a3-ghpc-[19-20]
#SBATCH --job-name=kuma-pretrain
#SBATCH --output=dmoe-log-pretrain.%j
#SBATCH --gpus-per-node=8
#SBATCH --mem=1800GB
#SBATCH --cpus-per-task=208

# インストールしたcondaを有効化。
source ~/miniconda3/etc/profile.d/conda.sh
which conda && echo "====" && conda --version
conda activate accessory 

# 分散環境の整備
# .sshディレクトリが存在するかどうかを確認
if [ ! -d ${HOME}/.ssh ]; then
  # 存在しない場合、ディレクトリを作成
  mkdir ${HOME}/.ssh
  echo ".ssh ディレクトリを作成しました。"
else
  echo ".ssh ディレクトリはすでに存在します。"
fi
# ベースラインコードによるssh環境の構築
bash ${HOME}/LLaMA2-Accessory/accessory/create_ssh_config_file_for_gcp_play_multi_node_multi_gpu.sh
#bash ${HOME}/ucllm_nedo_dev/train/scripts/common/create_ssh_config_file_for_gcp_play_multi_node_multi_gpu.sh 

sed -i "1i source ~/miniconda3/etc/profile.d/conda.sh" ~/.bashrc && \
    sed -i "2i conda activate accessory" ~/.bashrc && \
    sed -i "3i ulimit -n 262144 \n" ~/.bashrc

# シェルの実行
bash ${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/mixtral.sh
