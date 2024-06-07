#!/bin/bash

llama_config="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/config.json"
#tokenizer_path="${HOME}/tokenizer_model_directory"
tokenizer_path="/storage7/tokenizer_model_directory"
data_meta_path="${HOME}/LLaMA2-Accessory/data_example/Pretrain.json"
data_root="/storage7/pretraining_data/"

data_parallel=fsdp
model_parallel=2

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$((10000 + (RANDOM % 50000)))
export WORLD_SIZE=2
export RANK="${SLURM_PROCID}"
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
#export TOKENIZERS_PARALLELISM=true
ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"

ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="pretrain-Mixtral-dmoe-test"

exp_name="pretrain/Mixtral_wandb_test_multi"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

echo master_addr="$MASTER_ADDR" master_port="$MASTER_PORT" nnodes="$WORLD_SIZE" node_rank="$RANK"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" ${ucllm_nedo_dev}/accessory/main_pretrain.py \
--output_dir /storage7/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type mixtral_sparse --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--save_freq 5000 \
--max_words 4096 \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_test_multi" \
--wandb-project "${PROJECT_NAME}"
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

echo "exp name: $exp_name"
