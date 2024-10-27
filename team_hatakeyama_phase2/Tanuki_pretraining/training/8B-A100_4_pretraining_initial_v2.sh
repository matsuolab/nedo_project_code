#!/bin/bash

#checkpoint conversion
#/storage5/shared/hatakeyama/0611te/Megatron-LM/scripts/tsubame/ckpt/hf_to_megatron_llama3-8b_0627.sh

#conda activate 
# source ./miniconda3/etc/profile.d/conda.sh
# conda activate tanuki_pretraining_py310_TEv1.7_FAv2.5.7
export PYTHONPATH=/home/ubuntu/nedo_project_code/team_hatakeyama_phase2/Tanuki_pretraining/Megatron-LM:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# hostfile
export NUM_GPU_PER_NODE=4

model_size=8
NUM_LAYERS=32

HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # hiddensizeの3.5倍
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8

SEQ_LENGTH=2048

# distributed settings
TENSOR_PARALLEL_SIZE=1 
PIPELINE_PARALLEL_SIZE=4
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1536
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=1.0E-4
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=0.8

# model config
TOKENIZER_MODEL="./training/tokenizer/tokenizer_scale200.model"
CHECKPOINT_LOAD_DIR="./training/checkpoints/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"
CHECKPOINT_SAVE_DIR="./training/checkpoints/Llama-3-8/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

DATA_PATH_LIST=(
"./training/training_corpus/hanrei_1/combined_part_01_to_08_text_sentence"
"./training/training_corpus/hanrei_2/combined_part_18_to_25_text_sentence"
#"add more"
)

# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")

mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${log_path}

# checkpoint load
if [[ -f "${CHECKPOINT_LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
  resume training
 CHECKPOINT_ARGS="--load ${CHECKPOINT_LOAD_DIR}"
else
  first training
 CHECKPOINT_ARGS="--load ${CHECKPOINT_LOAD_DIR} --no-load-rng --no-load-optim"
fi

#finetune
# CHECKPOINT_ARGS="--load ${CHECKPOINT_LOAD_DIR} --finetune"

JOB_NAME="tanuki-8b"

# run
megatron_options="  \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_KEY_VALUE_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
	--tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 998,1,1 \
  --distributed-backend nccl \
  --init-method-std 0.008 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-05 \
  --log-interval 1 \
  --save-interval 5 \
  --eval-interval 10 \
  --eval-iters 2 \
  --bf16 \
  --use-checkpoint-args \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 10000.0 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --attention-softmax-in-fp32 \
  --recompute-activations \
  --recompute-granularity "selective" \
  --transformer-impl "transformer_engine" \
  --use-z-loss \
  --log-throughput \
	"
  # --fp8-format 'hybrid' \
  # --fp8-amax-compute-algo max \
  # --fp8-amax-history-len 1024 \
  # --no-fp8-wgrad \
  # --wandb-name ${JOB_NAME} \
  # --wandb-project "Llama-3-8B" \
  # --wandb-entity "weblab-geniac1" \

NNODES=1
NODE_RANK=0
# MASTER_ADDR=n_a
# MASTER_PORT=65400
DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPU_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK"

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

log_file="${log_path}/llma3_8B_${NODE_RANK}_${current_time}.log"

run_cmd="torchrun $DISTRIBUTED_ARGS ./Megatron-LM/pretrain_gpt.py
 ${megatron_options} \
 2>&1 | tee ${log_file}"

echo ${run_cmd}
eval ${run_cmd}
set +x