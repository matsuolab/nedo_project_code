#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

################
#Mistral v0.2 added 
#・rope_theta 1e6
#・No sliding window
#・Group query attention
#・Flash attantion

#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun

# Change the below configurations here
ucllm_nedo_dev_train_dir="/storage2/ucllm_nedo_prod/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# save_interval=1000
save_interval=200

output_model_dir="/storage2/output/3node"
# Modifies the arguments.
output_model_dir="${output_model_dir%/}"  # Removes a trailing slash "/" if it exists.

DATASET_1=/storage2/datasets/en_input/slimpajama_github_final_text_document
DATASET="1 ${DATASET_1}"
TOKENIZER_PATH="/storage2/tokenizer/en_ja_code_tokenizer/ja60K_en40K_code20K.model"
#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_TC=106
#echo "${NHOSTS}"

mp_size=1
pp_size=6
zero_stage=1
no_pp="false"
## Total number of GPUs.
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node="${NHOSTS}"
echo "num_node = ${num_node}"

num_gpus=$((${num_gpus_pernode} * ${num_node}))
## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

## Whether or not log optimizer states (norms, max, abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
seed=1234
num_workers=0
#MASTER_ADDR=localhost
#MASTER_PORT=6000
host="${HOSTNAME}"
NODE_RANK=${SLURM_NODEID}

#Llama 12.3B
HIDDEN_SIZE=5120 # e.g. mistral-7b: 4096
FFN_HIDDEN_SIZE=13824 #14336 # e.g. mistral-7b: 14337
NUM_LAYERS=40 # e.g. mistral-7b: 32 
NUM_HEADS=40 # e.g. mistral-7b: 3
NUM_KV_HEADS=8 # mistral-7b: 8
init_std=0.008

SEQ_LENGTH=2048 #: 32768
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1960 # e.g. llama: 4M tokens
rope_theta=1e5 #1e6
LR=3e-4
MIN_LR=3e-5
TRAIN_STEPS=250000
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

model_size=12

## Activation checkpointing saves GPU memory, but reduces training speed
#activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################

prescale_grad="true"
jobname="Llama2_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${MIN_LR}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_g${num_gpus}"
if [[ $zero_stage -gt 0 ]]; then
    jobname="${jobname}_z${zero_stage}"
    prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
    jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
    jobname="${jobname}_pp${pp_size}"
fi
jobname="${jobname}_seed${seed}_rebase"

username=$(whoami)
log_path="${output_model_dir}/log"
CHECKPOINT_PATH="${output_model_dir}/checkpoint/${jobname}"
tensorboard_path="${output_model_dir}/tensorboard/${jobname}_${host}_${current_time}"
deepspeed_config_dir="${output_model_dir}/deepspeed_config"
mkdir -p ${log_path}
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${tensorboard_path}
mkdir -p ${deepspeed_config_dir}
###############################################################################

config_json="${deepspeed_config_dir}/ds_config_gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_log${log_interval}_zero${zero_stage}.json"
template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"
sed "s/GBSIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/MBSIZE/${MICRO_BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
      > ${config_json}


cat <<EOT > $config_json
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $zero_stage
  },
  "bf16": {
    "enabled": true
  },
  "data_types": {
    "grad_accum_dtype": "fp32" 
  },  
  "wandb": {
    "enabled": true,    
    "project": "Llama3_3node",
    "group": "test"
  },
  "gradient_clipping": $GRAD_CLIP,
  "prescale_gradients": $prescale_grad,
  "wall_clock_breakdown": false
}
EOT

#
#  "wandb": {
#    "enabled": true,    
#    "project": "Llama3",
#    "group": "test"
#  },
#    "group": "GPU",

#    "team": "my_team"

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

data_options=" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --data-path ${DATASET} \
    --data-impl mmap"

exit_duration=300000000000

wandb_entity="test"
wandb_project="Llama3"

megatron_options=" \
    --num-workers ${num_workers} \
    --override-opt_param-scheduler \
    --tensor-model-parallel-size ${mp_size} \
    --pipeline-model-parallel-size ${pp_size} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --train-iters ${TRAIN_STEPS} \
    --save ${CHECKPOINT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --init-method-std ${init_std} \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --split 998,1,1 \
    --distributed-backend nccl \
    --lr ${LR} \
    --lr-decay-style cosine \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad ${GRAD_CLIP} \
    --lr-warmup-iters ${LR_WARMUP_STEPS} \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --log-interval 1 \
    --save-interval ${save_interval} \
    --eval-interval 200000 \
    --eval-iters 1000 \
    --bf16 \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --num-key-value-heads ${NUM_KV_HEADS} \
    --use-flash-attn-v2 \
    --seed ${seed} \
    --exit-duration-in-mins ${exit_duration} "

#--transformer-impl ${transformer_engine}

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

# Creates a hostfile.
script_dir=$(dirname "$0")
hostfile="${script_dir}/hostfile_jobid-${SLURM_JOB_ID}"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

for node in $nodes
do
  gpu_count=$(ssh ${node} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
  echo "${node} slots=${gpu_count}"
  ssh $node "source ~/.bashrc"
  ssh $node 'source /storage2/miniconda3/etc/profile.d/conda.sh && conda activate .venv'
done > "${hostfile}"

echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""

#megatron_options="${megatron_options} --distribute"

# Sets the master port number to a unique number.
master_port=$((10000 + (${SLURM_JOB_ID} % 50000)))

log_file="${log_path}/${jobname}_${host}_${current_time}_${SLURM_PROCID}.log"

deepspeed --master_port ${master_port} --hostfile ${hostfile} \
    ${megatron_deepspeed_dir}/pretrain_gpt.py \
    ${megatron_options} \
    ${data_options} \
    ${deepspeed_options} \
    2>&1 | tee "${log_file}"