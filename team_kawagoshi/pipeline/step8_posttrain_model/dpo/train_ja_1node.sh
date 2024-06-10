#!/bin/bash

set -e
echo ""

# Change the below configurations here
ucllm_nedo_dev_train_dir="/storage2/ucllm_nedo_prod/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"

echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
# Initializes the arguments.
input_model_name_or_path="/storage2/output/finetune/chat/checkpoint/SFT_zero1_2024.05.25_15.50.28/checkpoint-400"
OUTPUT_DIR="/storage2/output/finetune/dpo"
wandb_entity="weblab-geniac5"
wandb_project="DPO"
wandb_tag="multi"  # Optional argument.

mkdir -p ${OUTPUT_DIR}

current_time=$(date "+%Y.%m.%d_%H.%M.%S")
jobname="SFT_zero1_${current_time}"

username=$(whoami)
log_path="${OUTPUT_DIR}/log"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint/${jobname}"

mkdir -p ${log_path}
mkdir -p ${CHECKPOINT_PATH}

host="${HOSTNAME}"

wandb_options=" \
    --use_wandb \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project} \
    --wandb_group finetune_${input_model_name_or_path}_${current_time}"
if [[ -n "${wandb_tag}" ]]; then
wandb_options="${wandb_options} \
    --wandb_tag ${wandb_tag}"
fi

# Sets the master port number to a unique number.
master_port=$((10000 + (${SLURM_JOB_ID} % 50000)))

# Creates a hostfile.
script_dir=$(dirname "$0")
hostfile="${script_dir}/hostfile_jobid-${SLURM_JOB_ID}"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

for node in $nodes
do
  gpu_count=$(ssh ${node} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")

  echo "${node} slots=${gpu_count}"
done > "${hostfile}"

echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""

# Finetunes the pretrained model.
#deepspeed --master_port ${master_port} --hostfile ${hostfile} \
#accelerate launch --config_file /storage2/ucllm_nedo_prod/train/scripts/step4_finetune_model/pre_dev/deepspeed_config/accelerate_configs/zero2.yaml \    
deepspeed --master_port ${master_port} --hostfile ${hostfile} \
    ${ucllm_nedo_dev_train_dir}/llm-jp-dpo/train_speedup.py \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type constant_with_warmup \
    --bf16 \
    --model_name_or_path ${input_model_name_or_path} \
    --output_dir ${OUTPUT_DIR} \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 1 \
    --seed 42 \
    --precompute_ref_log_probs \
    --save_only_model \
    --use_flash_attention_2 \
    --deepspeed /storage2/ucllm_nedo_prod/train/scripts/step4_finetune_model/pre_dev/deepspeed_config/ds_config_zero3_nooffload.json \
    ${wandb_options} \
    2>&1 | tee ${log_path}/${host}_${current_time}.log

echo ""
echo "Finished to finetune the pretrained model."
echo ""
