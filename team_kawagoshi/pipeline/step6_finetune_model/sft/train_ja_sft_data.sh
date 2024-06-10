#!/bin/bash

set -e
echo ""

# Change the below configurations here
ucllm_nedo_dev_train_dir="/storage2/ucllm_nedo_prod/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"

echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
# Initializes the arguments.
input_model_name_or_path="/storage2/output/SFT/many_data/checkpoint/SFT_zero1_2024.05.22_07.35.09"
OUTPUT_DIR="/storage2/output/finetune/chat"
wandb_entity="weblab-geniac5"
wandb_project="SFT_chat "

# Prints the arguments.
echo "input_model_name_or_path = ${input_model_name_or_path}"
echo "output_dir = ${OUTPUT_DIR}"
echo "wandb_entity = ${wandb_entity}"
echo "wandb_project = ${wandb_project}"
echo ""

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
    --wandb_group finetune_${input_model_name_or_path}_${host}_${current_time}"

# Finetunes the pretrained model.

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
export CUDA_LAUNCH_BLOCKING=1

deepspeed --master_port ${master_port} --hostfile ${hostfile} \
    ${ucllm_nedo_dev_train_dir}/llm-jp-sft/train_ja.py \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --bf16 \
    --data_files original_finetuning_filtered-instructions-merged_v2.jsonl kawagoshi-llm-team-original_finetuning_multi-instructions-merged-sp.jsonl kawagoshi-llm-team-original_finetuning_math-instructions-merged.jsonl kawagoshi-llm-team-original_finetuing_multi_plus-instructions-merged-sp.jsonl additiona-multi-turn-instructions-merged-sp.jsonl additional-instructions-merged.jsonl \
    --sample_sizes 1 1 1 1 1 1 \
    --model_name_or_path ${input_model_name_or_path} \
    --output_dir ${CHECKPOINT_PATH} \
    --logging_steps 1 \
    --save_step 50 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --use_flash_attention_2 \
    --neftune_noise_alpha 10.0 \
    --valid_ratio 0.1 \
    --save_only_model \
    --deepspeed /storage2/ucllm_nedo_prod/train/scripts/step4_finetune_model/pre_dev/deepspeed_config/ds_config_zero1.json \
    ${wandb_options} \
    2>&1 | tee ${log_path}/${host}_${current_time}.log

echo ""
echo "Finished to finetune the pretrained model."
echo ""

# --sample_sizes 1 1 1 1 1 1 1 1000 \
# oasst2-33k-ja_converted-nosp.jsonl oasst1-21k-ja_converted-nosp.jsonl OpenMathInstruct-1-1.8m-ja-12000-converted-nosp.jsonl amenokaku-code-instruct-merged.jsonl HachiML-Evol-hh-rlhf-gen3-1k-merged.jsonl HachiML-Hachi-Alpaca-merged.jsonl kunishou-databricks-dolly-15k-ja-instructions-merged.jsonl AnswerCarefullyVersion001_Dev-merged.jsonl oasst2-33k-en_converted-nosp.jsonl

#, oasst1-21k-en-merged oasst2-33k-ja-merged oasst2-33k-en-merged 

#--data_files sudy-super-oasst2-chat-5k-ja-sp-filtered_v2.jsonl original_finetuning_filtered-instructions-merged_v2.jsonl kawagoshi-llm-team-original_finetuning_multi-instructions-merged-sp.jsonl oasst1-21k-ja_uniq_deduplication-merged-sp-filtered.jsonl kawagoshi-llm-team-original_finetuning_math-instructions-merged.jsonl kawagoshi-llm-team-original_finetuing_multi_plus-instructions-merged-sp.jsonl HachiML-alpaca_jp_python-instructions-merged.jsonl HachiML-alpaca_jp_math-instructions-merged.jsonl additiona-multi-turn-instructions-merged-sp.jsonl additional-instructions-merged.jsonl \
#--sample_sizes 1 1 1 1 1 1 1000 1000 1 1 \
    