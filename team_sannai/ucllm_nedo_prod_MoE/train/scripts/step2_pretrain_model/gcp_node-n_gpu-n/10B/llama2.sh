#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
# ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_dev/train"
ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_prod_MoE/train"
# megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed-MoE"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
# input_tokenizer_file="mistralai/Mixtral-8x7B-v0.1"
input_tokenizer_file="team-sanai/unigram_32000"
output_model_dir=""
save_interval=300
master_port=6008

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --output_model_dir) output_model_dir=${2}; shift ;;
        --save_interval) save_interval=${2}; shift ;;
        --master_port) master_port=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${output_model_dir} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --output_model_dir <output_model_dir>"
    exit 1
fi

# Modifies the arguments.
output_model_dir="${output_model_dir%/}"  # Removes a trailing slash "/" if it exists.

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "output_model_dir = ${output_model_dir}"
echo "save_interval = ${save_interval}"
echo ""

###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
# seq_len=2048
# seq_len=512
seq_len=4096

## The "GPT-3 XXX" below are configs from GPT-3 paper
## https://arxiv.org/abs/2005.14165, choose based on
## your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)

## We changed min_lr to a lower number (1.0e-6), which we found is able to
## provide better zero-shot eval results.

## GPT-3 TinyTiny (17.68M)
# model_size=0.01
# num_layers=6
# hidden_size=256
# num_attn_heads=4
# global_batch_size=128
# lr=6.0e-4
# min_lr=1.0e-6
# init_std=0.02

## llama2 Small 125M
model_size=0.125
num_layers=12
hidden_size=768
num_attn_heads=12
num_key_heads=4
ffn_hidden_size=768
#global_batch_size=256
global_batch_size=32
lr=6.0e-4
min_lr=1.0e-6
init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=256
# lr=3.0e-4
# min_lr=1.0e-6
# init_std=0.018

## llama2 Large 760M
model_size=0.76
num_layers=24
hidden_size=1530
num_attn_heads=16
num_key_heads=8
ffn_hidden_size=1536
# global_batch_size=256
global_batch_size=32
lr=2.5e-4
min_lr=1.0e-6
init_std=0.015

## llama2 915.62M
#model_size=7
## num_layers=32
num_layers=16
##hidden_size=4096
hidden_size=2560
#num_attn_heads=32
num_key_heads=8
ffn_hidden_size=11008
#global_batch_size=512
global_batch_size=1536
#global_batch_size=1024
lr=1.2e-4
min_lr=1.0e-6
init_std=0.009

## GPT-3 XL 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=16
# global_batch_size=512
# lr=2.0e-4
# min_lr=1.0e-6
# init_std=0.013

## GPT-3 2.7B
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=512
# lr=1.6e-4
# min_lr=1.0e-6
# init_std=0.011

## GPT-3 6.7B
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=1024
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

# ## llama2 7B
# model_size=7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# num_key_heads=32
# ffn_hidden_size=11008
# # global_batch_size=256
# global_batch_size=1536
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

## llama2 7B
model_size=7
num_layers=32
hidden_size=4096
# hidden_size=2560
num_attn_heads=32
num_key_heads=32
ffn_hidden_size=11008
# global_batch_size=256
# global_batch_size=1530
global_batch_size=1440
#global_batch_size=1080
#global_batch_size=180
lr=1.2e-4
min_lr=1.0e-6
init_std=0.009

# ## llama2 13B
# model_size=13
# num_layers=32
# hidden_size=5120
# num_attn_heads=40
# num_key_heads=40
# ffn_hidden_size=13824
# # global_batch_size=256
# global_batch_size=1536
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

## GPT-3 13B
# model_size=13
# num_layers=40
# hidden_size=5120
# num_attn_heads=40
# global_batch_size=1024
# lr=1.0e-4
# min_lr=1.0e-6
# init_std=0.008

## GPT-3 175B
# model_size=175
# num_layers=96
# hidden_size=12288
# num_attn_heads=96
# global_batch_size=1536
# lr=0.6e-4
# min_lr=1.0e-6
# init_std=0.005
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
# train_tokens_in_billion=300
# train_tokens_in_billion=3
# train_tokens_in_billion=1
# train_tokens_in_billion=2
# train_tokens=$((${train_tokens_in_billion} * 1000 * 1000 * 1000))

## train_samples is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
# train_samples=$(( 300 * 1000 * 1000 * 1000 * 2 / ${seq_len} ))
# train_samples=11197430 ## seed用のデータセットの行数
# train_samples=4199036 ## seed用のデータセット3, 4, 5の行数

train_iters=25000

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
# lr_warmup_tokens_in_million=3000
# lr_warmup_tokens_in_million=30
# lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000 * 1000))
# lr_warmup_tokens=${global_batch_size}*${seq_len}*2000
# lr_warmup_tokens=$((${global_batch_size}*${seq_len}*${lr_warmup_steps}))
## global_batch_size=64, seq_len=512のとき, 65536000(65M)

## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality 
# lr_decay_tokens_in_billion=${train_tokens_in_billion}
# lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000 * 1000 * 1000))
# lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
# mp_size=2
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
# pp_size=2
pp_size=4

# If you plan to use Megatron-DeepSpeed's deepspeed_to_transformers.py to convert
# the checkpoint from Megatron-DeepSpeed format to Hugging Face Transformers format,
# then sets no_pp to false (even if pp_size is 1).
# The reason why is because Megatron-DeepSpeed's deepspeed_to_transformers.py assumes
# there are "layer_*.pt" files, and "layer_*.pt" files are created if no_pp is false.
# In other words, if no_pp is true, then "layer_*.pt" files are not created and
# Megatron-DeepSpeed's deepspeed_to_transformers.py would fail.
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
# zero_stage=0
zero_stage=1

## Total number of GPUs.
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node="${NHOSTS}"
# num_node=2 ## NHOSTSがなかったとき用
num_node=3 ## NHOSTSがなかったとき用
# num_node=1
num_gpus=$((${num_gpus_pernode} * ${num_node}))

## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))
echo "dp,pp,mp=${dp_size},${pp_size},${mp_size}"

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
# batch_size=3
batch_size=2
###############################################################################
### Misc configs
# log_interval=100
log_interval=2
eval_iters=10
eval_interval=300
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
# num_save=100
# estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
num_workers=10

# # If either arxiv_text_document.bin or arxiv_text_document.idx doesn't exist yet,
# # then downloads arxiv.jsonl and preprocesses the data.
# data_path="${megatron_deepspeed_dir}/dataset/arxiv_text_document"
# if [ ! -f "${data_path}.bin" ] || [ ! -f "${data_path}.idx" ]; then
#     echo "Either ${data_path}.bin or ${data_path}.idx doesn't exist yet, so download arxiv.jsonl and preprocess the data."
#     wget https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl \
#         --directory-prefix ${megatron_deepspeed_dir}/dataset/
#     mv ${megatron_deepspeed_dir}/dataset/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl ${megatron_deepspeed_dir}/dataset/arxiv.jsonl
#     python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
#         --tokenizer-type SentencePieceTokenizer \
#         --tokenizer-model ${input_tokenizer_file} \
#         --input ${megatron_deepspeed_dir}/dataset/arxiv.jsonl \
#         --output-prefix ${megatron_deepspeed_dir}/dataset/arxiv \
#         --dataset-impl mmap \
#         --workers $(grep -c ^processor /proc/cpuinfo) \
#         --append-eod
# else
#     echo "Both ${data_path}.bin and ${data_path}.idx already exist."
# fi
# echo ""
data_path="${megatron_deepspeed_dir}/dataset/sample/wiki_text_document"

# data_path="/persistentshare/storage/team_sannai/fujimoto/seed_1G/dolma_c4_cc_mid_text_document"
# data_path="1 /storage6/test_corpus/seed_1G/dolma_c4_cc_mid_text_document"
# data_path="22.065075 /storage6/dataset/pretrain_converted/phase1/MATH/peS2o/peS2o_s2orcV2_en_sep_0.17_text_document"

data_path="
45.2136916 /storage6/dataset/pretrain_converted/phase1/1B/concat_ja_mc4/merged_text_document \
45.614523 /storage6/dataset/pretrain_converted/phase1/1B/en_refinedWeb/en_refinedWeb_1B_split_file_116.0_text_document \
21.2694435 /storage6/dataset/pretrain_converted/phase1/CODING/github_code_clean/github_code_clean_CODING_split_file_54.4_text_document \
13.9382895 /storage6/dataset/pretrain_converted/phase1/CODING/github_code_clean_python/github_code_clean_python_CODING_split_file_36.3_text_document \
1.3312965 /storage6/dataset/pretrain_converted/phase1/MATH/OpenWebMath/OpenWebMath_MATH_split_file_3.6_text_document \
22.065075 /storage6/dataset/pretrain_converted/phase1/MATH/peS2o/peS2o_s2orcV2_en_sep_0.17_text_document
"

data_path="
10 /storage6/dataset/pretrain_converted/phase1/1B/en_refinedWeb/en_refinedWeb_1B_split_file_116.0_text_document \
30 /storage6/dataset/pretrain_converted/phase1/1B/concat_ja_mc4/merged_text_document \
1.75 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean/github_code_clean_00000_of_00880_text_document \
1.75 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean/github_code_clean_00001_of_00880_text_document \
1.75 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean/github_code_clean_00002_of_00880_text_document \
1.75 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean/github_code_clean_00003_of_00880_text_document \
3 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean_python/github_code_clean_python_00001_text_document \
3 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean_python/github_code_clean_python_00002_text_document \
3 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean_python/github_code_clean_python_00003_text_document \
3 /storage6/dataset/pretrain_converted_v02/other/CODING/raw/github_code_clean_python/github_code_clean_python_00004_text_document \
0.5 /storage6/dataset/pretrain_converted_v02/other/MATH/raw/OpenWebMath/openwebmath-1_text_document \
0.5 /storage6/dataset/pretrain_converted_v02/other/MATH/raw/OpenWebMath/openwebmath-2_text_document \
3 /storage6/dataset/pretrain_converted_v02/other/MATH/raw/OpenMathInstruct/ja_math_text_document \
1 /storage6/dataset/pretrain_converted_v02/other/MATH/raw/StackMath/StackMathQA_en_text_document \
10 /storage6/dataset/pretrain_converted/phase1/MATH/peS2o/peS2o_s2orcV2_en_sep_0.17_text_document \
0.4 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_en/wikipedia_en_00000_of_00033_text_document \
0.4 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_en/wikipedia_en_00001_of_00033_text_document \
0.4 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_en/wikipedia_en_00002_of_00033_text_document \
0.4 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_en/wikipedia_en_00003_of_00033_text_document \
0.4 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_en/wikipedia_en_00003_of_00033_text_document \
1 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_ja/wikipedia_ja_00000_of_00008_text_document \
1 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_ja/wikipedia_ja_00001_of_00008_text_document \
1 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_ja/wikipedia_ja_00002_of_00008_text_document \
1 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_ja/wikipedia_ja_00003_of_00008_text_document \
1 /storage6/dataset/pretrain_converted_v02/other/WIKI/raw/wikipedia_ja/wikipedia_ja_00004_of_00008_text_document \
4 /storage6/aa_fujimoto/bin_dataset/sft_en_text_document \
4 /storage6/aa_fujimoto/bin_dataset/sft_ja_text_document \
2 /storage6/aa_fujimoto/bin_dataset/ultra_text_text_document \
5 /storage6/dataset/pretrain_converted_v02/other/BOOK/raw/JA/aozora/ja_book_text_document
"


prescale_grad="true"
# jobname="gpt_${model_size}B_tok${train_tokens_in_billion}B"
jobname="gpt_${model_size}B"
# jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_lr${lr}_min${min_lr}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
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
    --data-path ${data_path} \
    --data-impl mmap"

## use HF tokenizer
data_options=" \
    --tokenizer-type HFTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --data-path ${data_path} \
    --data-impl mmap \
    --data-cache-path /storage6/aa_fujimoto/llama2_data_index
"    
#--data-cache-path ./llama2_data_index
#"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --split 949,50,1 \
    --lr ${lr} \
    --clip-grad 1.0 \
    --log-interval ${log_interval} \
    --save-interval ${save_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --fp16 \
    --seed ${seed} \
    --tensorboard-dir ${tensorboard_path} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --override-opt_param-scheduler \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --exit-duration-in-mins ${exit_duration} \
    --use-flash-attn-v2 \
    --train-iters $train_iters \
    --weight-decay 0.1 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --no-async-tensor-model-parallel-allreduce"

## llama2
megatron_options="${megatron_options} \
    --optimizer adam \
    --num-key-value-heads ${num_key_heads} \
    --ffn-hidden-size ${ffn_hidden_size} \
    --swiglu \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --normalization rmsnorm \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear"

## default scheduler
# GPT_ARGS="${GPT_ARGS} \
#     --lr-warmup-fraction .01 \
#     --lr-decay-iters $ITERATION \
#     --min-lr ${MIN_LR} \
#     --lr-decay-style cosine"

## stable_decay_scheduler
LR_WARMUP_ITERS=1
STABLE_TRAIN_ITERS=8900
WSD_T=3700
megatron_options="${megatron_options} \
    --lr-warmup-iters ${LR_WARMUP_ITERS}\
    --use-wsd \
    --stable-train-iters ${STABLE_TRAIN_ITERS} \
    --wsd-tmp ${WSD_T}    
"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

## config
config_json="${deepspeed_config_dir}/ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
# template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"
template_json="./ds_config_gpt_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
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

# Creates a hostfile.
script_dir=$(dirname "$0")
hostfile="${script_dir}/hostfile_jobid-${SLURM_JOB_ID}"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

for node in $nodes
do
  #gpu_count=$(ssh ${node} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
  gpu_count=8
  echo "${node} slots=${gpu_count}"
done > "${hostfile}"

echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""


#MASTER_ADDR=localhost
MASTER_ADDR=172.16.0.9
# MASTER_PORT=6001

# DISTRIBUTED_ARGS="
#     --master_addr ${MASTER_ADDR} \
#     --master_port ${master_port}
# "
DISTRIBUTED_ARGS="
    --master_addr ${MASTER_ADDR} \
    --master_port ${master_port} \
    --hostfile ${hostfile}
"


deepspeed ${DISTRIBUTED_ARGS} ${megatron_deepspeed_dir}/pretrain_gpt.py \
    ${megatron_options} \
    ${data_options} \
    ${deepspeed_options} \
    2>&1 | tee ${log_path}/${jobname}_${host}_${current_time}.log
