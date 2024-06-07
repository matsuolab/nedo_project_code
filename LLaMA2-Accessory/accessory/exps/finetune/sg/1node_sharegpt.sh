#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=a3
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=360GB
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=slurm0-a3-ghpc-[18-20]
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
conda activate finetune


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

#llama_config="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/config.json"
pretrained_type=consolidated
tokenizer_path="/storage7/tokenizer_model_directory"
data_config="${HOME}/LLaMA2-Accessory/accessory/configs/data/finetune/sg/dialog_sharegpt.yaml"

data_parallel=sdp
model_parallel=4

ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"

ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="finetune-Mixtral-dmoe"

exp_name="finetune/Mixtral_3node_finetune_single_multi_v3"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
export WORLD_SIZE=1

srun torchrun --nproc_per_node 8 \
--nnodes="$WORLD_SIZE" ${ucllm_nedo_dev}/accessory/main_finetune.py \
--output_dir /storage7/"$exp_name" \
--epochs 3 --warmup_epochs 0.04 \
--batch_size 8 --accum_iter 24 --num_workers 8 \
--lr 0.00002 --min_lr 0 --clip_grad 2 --weight_decay 0.0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type mixtral_sparse --tokenizer_path "$tokenizer_path" \
--no_visual \
--precision bf16 \
--max_words 2048 \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_3B_1node" \
--wandb-project "${PROJECT_NAME}" \
--data_config $data_config \
--dialog \
--pretrained_path /storage7/finetune/Mixtral_3node_finetune_single_v3/epoch2 \
--pretrained_type="$pretrained_type"
