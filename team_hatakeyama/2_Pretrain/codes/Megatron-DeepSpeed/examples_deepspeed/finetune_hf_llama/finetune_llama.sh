HF_LLAMA_PATH=/storage5/llm/codes/2_pretrain/mujina_test

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TP=1
PP=3
# require to align with weight dimensions
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_LAYERS=48
NUM_HEADS=32
SEQ_LENGTH=8192
######################################

MEGA_DS_LLAMA_PATH=/storage5/llm/codes/2_pretrain/mujina_test_megads

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

covert_args="deepspeed /storage5/llm/codes/2_pretrain/Megatron-DeepSpeed/tools/hf2megads_weight_converter.py \
--hf-ckpt-num-shards 9 \
--origin-hf-ckpt-dir $HF_LLAMA_PATH \
--save $MEGA_DS_LLAMA_PATH"

finetune_args="deepspeed finetune_llama.py \
--load $MEGA_DS_LLAMA_PATH"

comm_args=" \
    --finetune \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --tensor-model-parallel-size 1 \
    --init-method-std 0.02 \
    --micro-batch-size 1 \
    --exit-duration-in-mins 300000000000 \
    --global-batch-size 1024 \
    --num-layers 48 \
    --hidden-size 4096 \
    --save-interval 100 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --max-position-embeddings 8192 \
    --distributed-backend nccl \
    --split 949,50,1 \
    --eval-interval 50 \
    --eval-iters 5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --num-workers 200 \
    --tokenizer-type HFTokenizer \
    --tokenizer-model $HF_LLAMA_PATH \
    --deepspeed_config /storage5/llm/codes/2_pretrain/Megatron-DeepSpeed/examples_deepspeed/rebase/ds_config_fix.json \
    --deepspeed \
    --bf16 \
    --seed 42 \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn-v2 \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --ds-sequence-parallel-size 1 \
    --use-pin-memory \
    --no-query-key-layer-scaling \
    --layernorm-epsilon 1e-5 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --no-bias-gelu-fusion \
    --no-masked-softmax-fusion \
    --no-position-embedding \
    --num-key-value-heads 8 \
    --recompute-activations \
    --recompute-granularity 'selective'"

task_args="$covert_args"

full_cmd="$task_args $comm_args"

eval "$full_cmd"

