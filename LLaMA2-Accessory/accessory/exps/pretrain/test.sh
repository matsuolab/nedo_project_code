#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=a3
#SBATCH --time=72:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=slurm0-a3-ghpc-[18-19]
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=26
#SBATCH --output=/storage7/nccl_test_%j.out

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=/storage7/nccl_debug_${SLURM_JOB_ID}_${SLURM_NODEID}.log
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl2/lib:/lib


llama_config="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/config.json"
#tokenizer_path="${HOME}/tokenizer_model_directory"
tokenizer_path="/storage7/tokenizer_model_directory"
data_meta_path="${HOME}/LLaMA2-Accessory/data_example/Pretrain1.json"
data_root="/storage7/pretraining_data/"

data_parallel=fsdp
model_parallel=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=8964
export WORLD_SIZE=2
export RANK="${SLURM_PROCID}"
#export OMP_NUM_THREADS=1
#export TOKENIZERS_PARALLELISM=true
ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"

ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="pretrain-Mixtral-dmoe"

exp_name="pretrain/Mixtral_wandb_multi_18"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

echo master_addr="$MASTER_ADDR" master_port="$MASTER_PORT" nnodes="$WORLD_SIZE" node_rank="$RANK"

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

#torchrun --nproc_per_node=2 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
#--log-dir="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/log" \
#--nnodes=1:2 --node_rank="$RANK" 
srun --output=0 python ${ucllm_nedo_dev}/accessory/main_pretrain.py \
--output_dir /storage7/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 50000 --lr_decay_iters 4000000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type mixtral_sparse --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--save_freq 20000 \
--max_words 4096 \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_multi_first" \
--wandb-project "${PROJECT_NAME}"
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

echo "exp name: $exp_name"
