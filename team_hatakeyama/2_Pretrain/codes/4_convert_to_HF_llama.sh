#!/bin/bash

#llama系モデルを変換するscript

set -e
echo "begin converting.."

# Stores the directory paths as variables.
megatron_deepspeed_dir=$(yq -r '.megatron_deepspeed_dir' config.yaml)
temp_dir=$(yq -r '.temp_dir' convert_config.yaml)
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""


input_tokenizer_file=$(yq -r '.input_tokenizer_file' config.yaml)
input_model_dir=$(yq -r '.input_model_dir' convert_config.yaml)
output_tokenizer_and_model_dir=$(yq -r '.output_tokenizer_and_model_dir' convert_config.yaml)


# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_model_dir = ${input_model_dir}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}

echo "converting tokenizers"
# Converts the tokenizer from SentencePiece format to HuggingFace Transformers format.
#python convert_tokenizer_from_sentencepiece_to_huggingface_transformers.py \
#    --input_tokenizer_file ${input_tokenizer_file} \
#    --output_tokenizer_dir ${output_tokenizer_and_model_dir}
#echo ""

python convert_llmjp_tokenizer.py \
    -i ${input_tokenizer_file} \
    -o ${output_tokenizer_and_model_dir}
echo ""

# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
#python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_to_transformers.py \
#    --input_folder ${input_model_dir} \
#    --output_folder ${output_tokenizer_and_model_dir}

rm -rf ${temp_dir}
mkdir ${temp_dir}

echo "converting to megatron"
python Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_megatron.py --input_folder ${input_model_dir} --output_folder ${temp_dir}

echo "converting to HF"
python llama_checkpoint_conversion.py --load_path ${temp_dir} --save_path ${output_tokenizer_and_model_dir} --convert_checkpoint_from_megatron_to_transformers --model_name Llama2

echo "Finished to convert the tokenizer and the pretrained model to HuggingFace Transformers format."

