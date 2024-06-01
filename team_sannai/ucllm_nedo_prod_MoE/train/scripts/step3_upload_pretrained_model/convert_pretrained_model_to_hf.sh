#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_prod_MoE/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed-MoE"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_model_dir=""
output_model_dir=""

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_model_dir) input_model_dir=${2}; shift ;;
        --output_model_dir) output_model_dir=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_model_dir} ]] || [[ -z ${output_model_dir} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_model_dir <input_model_dir> --output_model_dir <output_model_dir>"
    exit 1
fi

# Prints the arguments.
echo "input_model_dir = ${input_model_dir}"
echo "output_model_dir = ${output_model_dir}"
echo ""

mkdir -p ${output_model_dir}


# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_gpt2_to_hf.py.py \
    --input_folder ${input_model_dir} \
    --output_folder ${output_model_dir}

echo ""
echo "Finished to convert the tokenizer and the pretrained model to HuggingFace Transformers format."
echo ""
