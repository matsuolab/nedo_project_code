#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_dev/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_tokenizer_file=""
input_data_path=""

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --input_data_path) input_data_path=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${input_data_path} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --input_data_path <input_data_path>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_data_path = ${input_data_path}"
echo ""


data_path="${input_data_path}_text_document"
if [ ! -f "${data_path}.bin" ] || [ ! -f "${data_path}.idx" ]; then
    echo "Either ${data_path}.bin or ${data_path}.idx doesn't exist yet, so preprocess the data."
    python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model ${input_tokenizer_file} \
        --input ${input_data_path}.jsonl \
        --output-prefix ${input_data_path} \
        --dataset-impl mmap \
        --workers $(grep -c ^processor /proc/cpuinfo) \
        --append-eod
else
    echo "Both ${data_path}.bin and ${data_path}.idx already exist."
    echo -n "Do you want to reprocess the file? (y/n): "
    read answer

    if [[ "$answer" == [Yy]* ]]; then
        echo "Reprocessing the data..."
        python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model ${input_tokenizer_file} \
        --input ${input_data_path}.jsonl \
        --output-prefix ${input_data_path} \
        --dataset-impl mmap \
        --workers $(grep -c ^processor /proc/cpuinfo) \
        --append-eod
    else
        echo "Skipping reprocessing."
    fi

fi
echo ""