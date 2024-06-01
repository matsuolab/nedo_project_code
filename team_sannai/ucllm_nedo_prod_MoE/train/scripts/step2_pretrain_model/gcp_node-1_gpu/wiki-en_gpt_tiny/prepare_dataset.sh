#!/bin/bash

# sample...
# jsonl_file="./train.jsonl"
# output_prefix=${megatron_deepspeed_dir}"/dataset/sample/wiki"
# tokenizer="microsoft/phi-2"

# Initializes the arguments.
jsonl_file=""
output_prefix=""
tokenizer="mistralai/Mixtral-8x7B-v0.1"

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --jsonl_file) jsonl_file=${2}; shift ;;
        --output_prefix) output_prefix=${2}; shift ;;
        --tokenizer) tokenizer=${3}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${jsonl_file} ]] || [[ -z ${output_prefix} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --jsonl_file <jsonl_file> --output_prefix <output_prefix>"
    exit 1
fi

ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_prod_MoE/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed-MoE"
mkdir -p ${megatron_deepspeed_dir}"/dataset/sample"

tokenizer_type="SentencePieceTokenizer"
tokenizer_type="HFTokenizer"

python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
    --tokenizer-type ${tokenizer_type} \
    --tokenizer-model ${tokenizer} \
    --input ${jsonl_file} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers $(grep -c ^processor /proc/cpuinfo) \
    --append-eod
