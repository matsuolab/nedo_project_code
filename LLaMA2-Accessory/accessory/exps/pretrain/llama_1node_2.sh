#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=a3
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=1800GB
#SBATCH --cpus-per-task=26
#SBATCH --nodelist=slurm0-a3-ghpc-[18-19]
#SBATCH --output=/storage7/log/nccl_test_%j.out

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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate accessory


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export LOGLEVEL=INFO

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=/storage7/log/nccl_debug_${SLURM_JOB_ID}_${SLURM_NODEID}.log
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl2/lib:/lib


llama_config="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/config.json"
#tokenizer_path="${HOME}/tokenizer_model_directory"
tokenizer_path="/storage7/tokenizer_model_directory"
data_meta_path="${HOME}/LLaMA2-Accessory/data_example/Pretrain1.json"
data_root="/storage7/pretraining_data"

data_parallel=sdp
model_parallel=2

ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"

ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="LLama2"

exp_name="pretrain/Mixtral_wandb_multi_18"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
export WORLD_SIZE=1

srun torchrun --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
--nnodes="$WORLD_SIZE" ${ucllm_nedo_dev}/accessory/main_pretrain.py \
--output_dir /storage7/"$exp_name" \
--batch_size 4 --accum_iter 1 --num_workers 4 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 50 --lr_decay_iters 2000000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--save_freq 20000 \
--max_words 2048 \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_3B_2node_LLama2" \
--wandb-project "${PROJECT_NAME}"
