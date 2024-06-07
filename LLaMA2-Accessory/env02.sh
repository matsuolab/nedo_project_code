#!/bin/bash

# Command line options go here
#SBATCH --partition=g2         		 	 # 使用するパーティション名
#SBATCH --time=06:00:00         		 # 最大実行時間（例: 1分） (松尾研ルール: 00:00:00 - 06:00:00)
#SBATCH --nodes=1               		 # ノードの数
#SBATCH --ntasks-per-node=1     		 # ノードあたりのタスク数（通常は1） (松尾研ルール: 1)
#SBATCH --job-name=jk_environ-job 		 # ジョブ名
#SBATCH --output=jk_environ-job.out	 # 標準出力とエラーの出力ファイル名
#SBATCH --gpus-per-node=0 		  	 # ノードあたりのGPU数 (松尾研ルール: 0 - 1) 0ならCPUのみ利用する

# 既存環境削除
# Step 0-2. Python仮想環境の作成
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
pip install --upgrade pip setuptools wheel
cd ~/LLaMA2-Accessory
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install megablocks
which python && echo "====" && python --version
echo $PATH
echo $LD_LIBRARY_PATH
echo "#################### step0-2 OK ####################"
# Step 0-3. apexのインストール
cd ~/apex
pip uninstall ninja -y && pip install ninja==1.11.1
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
echo "#################### step0-4 OK ####################"
cd ~/LLaMA2-Accessory
pip install -e .
pip install pyarrow fastparquet
