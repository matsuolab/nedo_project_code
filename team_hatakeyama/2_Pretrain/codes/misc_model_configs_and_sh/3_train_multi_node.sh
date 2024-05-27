#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh && conda activate .venv_train

set -e
echo "load settings..."


# Stores the directory paths as variables.
megatron_deepspeed_dir=$(yq -r '.megatron_deepspeed_dir' multinode_config.yaml)
input_tokenizer_file=$(yq -r '.input_tokenizer_file' multinode_config.yaml)
tokenized_data_path=$(yq -r '.tokenized_data_path' multinode_config.yaml)
output_model_dir=$(yq -r '.output_model_dir' multinode_config.yaml)
save_interval=$(yq -e '.save_interval' multinode_config.yaml)
# Prints the arguments.
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "output_model_dir = ${output_model_dir}"
echo "save_interval = ${save_interval}"
echo ""


model_size=$(yq -e '.model_size' multinode_config.yaml)
num_layers=$(yq -e '.num_layers' multinode_config.yaml)
hidden_size=$(yq -e '.hidden_size' multinode_config.yaml)
num_attn_heads=$(yq -e '.num_attn_heads' multinode_config.yaml)
global_batch_size=$(yq -e '.global_batch_size' multinode_config.yaml)
lr=$(yq -e '.lr' multinode_config.yaml)
min_lr=$(yq -e '.min_lr' multinode_config.yaml)
init_std=$(yq -e '.init_std' multinode_config.yaml)
seq_len=$(yq -e '.seq_len' multinode_config.yaml)


echo "Model Size: $model_size"
echo "Number of Layers: $num_layers"
echo "Hidden Size: $hidden_size"
echo "Number of Attention Heads: $num_attn_heads"
echo "Global Batch Size: $global_batch_size"
echo "Learning Rate: $lr"
echo "Minimum Learning Rate: $min_lr"
echo "Init Std: $init_std"
echo "Seq len: $seq_len"
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=300
train_tokens=$((${train_tokens_in_billion} * 1000 * 1000 * 1000))


#1 epoch程度になるようにtoken数を決める
train_tokens=$(yq -e '.train_tokens' multinode_config.yaml)

#ここを適当に大きくしすぎると､必要メモリが増えすぎるので注意｡
##30000*...とかにすると､RAMが600GB必要､みたいになる
#train_samples=$(( 300 * 1000 * 1000 * 1000 * 2 / ${seq_len} ))

train_samples=$(yq -e '.train_samples' multinode_config.yaml)

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
exit_duration=300000000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
lr_warmup_tokens_in_million=2000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000 * 1000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality 
## lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens_in_billion=$(yq -e '.lr_decay_tokens_in_billion' multinode_config.yaml)
lr_decay_tokens=$(yq -e '.lr_decay_tokens' multinode_config.yaml)
lr_decay_style="cosine"

echo "lr_decay tokens, $lr_decay_tokens"

###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=$(yq -e '.mp_size' multinode_config.yaml)

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=$(yq -e '.pp_size' multinode_config.yaml)

# If you plan to use Megatron-DeepSpeed's deepspeed_to_transformers.py to convert
# the checkpoint from Megatron-DeepSpeed format to Hugging Face Transformers format,
# then sets no_pp to false (even if pp_size is 1).
# The reason why is because Megatron-DeepSpeed's deepspeed_to_transformers.py assumes
# there are "layer_*.pt" files, and "layer_*.pt" files are created if no_pp is false.
# In other words, if no_pp is true, then "layer_*.pt" files are not created and
# Megatron-DeepSpeed's deepspeed_to_transformers.py would fail.
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=$(yq -e '.zero_stage' multinode_config.yaml)

## Total number of GPUs.
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#num_gpus_pernode=1
#NHOSTS=1

NHOSTS=$(yq -e '.NHOSTS' multinode_config.yaml)
num_node="${NHOSTS}"
num_gpus=$((${num_gpus_pernode} * ${num_node}))
## Data parallel size.
sp_size=$(yq -e '.sp_size' multinode_config.yaml)
## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} /${sp_size}))
#dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

echo "num_gpus_pernode = ${num_gpus_pernode}"
echo "num_node = ${num_node}"
echo "num_gpus = ${num_gpus}"
echo "dp_size = ${dp_size}"

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
#batch_size=$(( ${global_batch_size} / ${dp_size} ))

micro_batch_size=$(yq -e '.micro_batch_size' multinode_config.yaml)
# batch_size=2
###############################################################################
### Misc configs
log_interval=1
eval_iters=10
eval_interval=46875000 # evalでtimeoutしてそうなので、外す
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
#activation_checkpoint="false"
activation_checkpoint=$(yq -r '.activation_checkpoint' multinode_config.yaml)

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234

num_workers=$(yq -e '.num_workers' multinode_config.yaml)

prescale_grad="true"
jobname="gpt_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${micro_batch_size}_g${num_gpus}"
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
checkpoint_path="${output_model_dir}/checkpoint/${jobname}"
tensorboard_path="${output_model_dir}/tensorboard/${jobname}_${host}_${current_time}"
deepspeed_config_dir="${output_model_dir}/deepspeed_config"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
mkdir -p ${deepspeed_config_dir}

###############################################################################
data_options=" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --data-path ${tokenized_data_path} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${micro_batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --bf16 \
    --seed ${seed} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn-v2 \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --ds-sequence-parallel-size ${sp_size} \
    --use-pin-memory \
    --ffn-hidden-size 17920 \
    --tensorboard-dir ${tensorboard_path}"

#    --use-rotary-position-embeddings \
#--transformer-impl transformer_engine \
if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

config_json="${deepspeed_config_dir}/ds_config_gbs${global_batch_size}_mbs${micro_batch_size}_log${log_interval}_zero${zero_stage}.json"
template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${micro_batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
      > ${config_json}

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

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
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

echo creating host file

# Creates a hostfile.
#script_dir=$(dirname "$0")
script_dir=/storage5/llm/codes/2_pretrain
hostfile="${script_dir}/hostfile_jobid-${SLURM_JOB_ID}"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

echo $nodes
for node in $nodes
do
  ssh $node "source ~/.bashrc"
  ssh $node 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate .venv_train'
  gpu_count=$(ssh ${node} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
  echo "${node} slots=${gpu_count}"
done > "${hostfile}"

echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""


echo "${megatron_options}"
wandb_options=" \
    --use_wandb \
    --wandb_project pretrain \
    --wandb_group pretrain_gpt_${model_size}B_${host}_${current_time}"

deepspeed --hostfile ${hostfile} \
       ${megatron_deepspeed_dir}/pretrain_gpt.py \
       ${megatron_options} \
       ${data_options} \
       ${deepspeed_options} \
        2>&1 | tee ${log_path}/${jobname}_${host}_${current_time}.log
