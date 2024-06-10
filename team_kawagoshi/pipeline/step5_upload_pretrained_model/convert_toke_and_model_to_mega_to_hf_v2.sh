#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
# ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_dev/train"
ucllm_nedo_dev_train_dir="/storage2/ucllm_nedo_prod/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"
llm_jp_tokeninzer_dir="${ucllm_nedo_dev_train_dir}/llm-jp-tokenizer"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_tokenizer_file=""
input_model_dir=""
output_tokenizer_and_model_dir=""
temp_model_dir=""
model_name=""
target_tp=1
target_pp=1

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --input_model_dir) input_model_dir=${2}; shift ;;
        --output_tokenizer_and_model_dir) output_tokenizer_and_model_dir=${2}; shift ;;
        --temp_model_dir) temp_model_dir=${2}; shift ;;
        --model_name) model_name=${2}; shift ;;
        --target_tp) target_tp=${2}; shift ;;
        --target_pp) target_pp=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${input_model_dir} ]] || [[ -z ${output_tokenizer_and_model_dir} ]] || [[ -z ${model_name} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --input_model_dir <input_model_dir> --output_tokenizer_and_model_dir <output_tokenizer_and_model_dir> --moel_name <model_name>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_model_dir = ${input_model_dir}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo "temp_model_dir = ${temp_model_dir}"
echo "model_name = ${model_name}"
echo "target_tp = ${target_tp}"
echo "target_pp = ${target_pp}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}
mkdir -p ${temp_model_dir}

# Converts the tokenizer from SentencePiece format to HuggingFace Transformers format.
python  ${llm_jp_tokeninzer_dir}/hf/convert_llmjp_unigram_spm_to_hf_fast.py \
    -i ${input_tokenizer_file} \
    -o ${output_tokenizer_and_model_dir} 
# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_to_megatron.py \
    --input_folder ${input_model_dir} \
    --output_folder ${temp_model_dir} \
    --target_pp ${target_pp} \
    --target_tp ${target_tp}

python ${ucllm_nedo_dev_train_dir}/scripts/step3_upload_pretrained_model/llama_checkpoint_conversion.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --save_path ${output_tokenizer_and_model_dir} \
    --print-checkpoint-structure \
    --megatron-path ${ucllm_nedo_dev_train_dir}/train/Megatron-DeepSpeed/megatron/ \
    --load_path ${temp_model_dir}   \
    --model_name ${model_name}
   
echo ""
echo "Finished to convert the tokenizer and the pretrained model to HuggingFace Transformers format."
echo ""
