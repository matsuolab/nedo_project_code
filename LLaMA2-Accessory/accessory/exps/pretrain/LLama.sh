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


tokenizer_path="/storage7/tokenizer_model_directory"
data_meta_path="${HOME}/LLaMA2-Accessory/data_example/test.json"
data_root="/storage7/test_data/"
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$((10000 + (RANDOM % 50000)))
export WORLD_SIZE=1
export RANK="${SLURM_PROCID}"
data_parallel=fsdp
model_parallel=2
exp_name="pretrain/13B_packed"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"
ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="LLama2"
echo master_addr="$MASTER_ADDR" master_port="$MASTER_PORT" nnodes="$WORLD_SIZE" node_rank="$RANK"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" ${ucllm_nedo_dev}/accessory/main_pretrain.py \
--output_dir /storage7/"$exp_name" \
--batch_size 8 --accum_iter 16 --num_workers 4 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_1node_llama" \
--wandb-project "${PROJECT_NAME}"
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

echo "exp name: $exp_name"
