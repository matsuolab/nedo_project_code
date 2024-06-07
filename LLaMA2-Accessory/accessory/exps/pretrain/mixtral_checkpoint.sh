#!/bin/bash

exp_name="pretrain/Mixtral_wandb_test"
iter_num="49"
#llama_config="${HOME}/LLaMA2-Accessory/accessory/exps/pretrain/config.json"
tokenizer_path="${HOME}/tokenizer_model_directory"
data_meta_path="${HOME}/LLaMA2-Accessory/data_example/Pretrain.json"
data_root="/persistentshare/storage/team_kumagai/datasets/model_data/wiki_parquet/"
checkpoint_path="/persistentshare/storage/team_kumagai/model/${exp_name}/epoch0-iter${iter_num}"
echo "$checkpoint_path"
data_parallel=fsdp
model_parallel=8

export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$((10000 + (RANDOM % 50000)))
export WORLD_SIZE=1
export RANK=0
#export TOKENIZERS_PARALLELISM=true
ucllm_nedo_dev="${HOME}/LLaMA2-Accessory"

ENTITY_NAME="weblab-geniac2"
PROJECT_NAME="pretrain-Mixtral-dmoe-pre"

echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

echo master_addr="$MASTER_ADDR" master_port="$MASTER_PORT" nnodes="$WORLD_SIZE" node_rank="$RANK"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" ${ucllm_nedo_dev}/accessory/main_pretrain.py \
--output_dir /persistentshare/storage/team_kumagai/model/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type mixtral_sparse --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--save_freq 5000 \
--max_words 4096 \
--resume "$checkpoint_path" \
--pretrained_path "$checkpoint_path" \
--pretrained_type consolidated \
--wandb-entity "${ENTITY_NAME}" \
--wandb-name "wandb_test_${iter_num}" \
--wandb-project "${PROJECT_NAME}"
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

echo "exp name: $exp_name"
