#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate accessory

llama_config="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/config.json"
#tokenizer_path="${HOME}/tokenizer_model_directory"
tokenizer_path="/storage7/tokenizer_model_directory"
data_meta_path="${HOME}/LLaMA2-Accessory/data_example/test.json"
data_root="/storage7/test_data/"

data_parallel=fsdp
model_parallel=8

ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"

ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="pretrain-Mixtral-dmoe"

exp_name="pretrain/Mixtral_wandb_multi_18"
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node 8 \
--master_addr=10.0.0.12 --master_port=29500 \
--nnodes=2 --node_rank=1 ${ucllm_nedo_dev}/accessory/main_pretrain.py \
--output_dir /storage7/"$exp_name" \
--batch_size 12 --accum_iter 1 --num_workers 8 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 2000000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type mixtral_sparse --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--save_freq 20000 \
--max_words 2048 \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_3B_2node_%j" \
--wandb-project "${PROJECT_NAME}"
