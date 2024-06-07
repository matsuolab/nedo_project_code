#!/bin/bash


# Command line options go here
#SBATCH --partition=g2                                                  # 使用するパーティション名
#SBATCH --time=02:00:00                                                 # 最大実行時間（例: 1分） (松尾研ルール: 00:00:00 - 06:00:00)
#SBATCH --nodes=1                                                       # ノードの数
#SBATCH --ntasks-per-node=1                                             # ノードあたりのタスク数（通常は1） (松尾研ルール: 1)
#SBATCH --job-name=merge_and_convert_and_upload_MegaBlocks                # ジョブ名
#SBATCH --output=log/%x.out                                         # 標準出力とエラーの出力ファイル名
#SBATCH --gpus-per-node=1                                               # ノードあたりのGPU数 (松尾研ルール: 0 - 1) 0ならCPUのみ利用する
#SBATCH --cpus-per-task=4


source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv


model_dir="share_dir/output_mixtral-dmoe"
ckpt_dir="${model_dir}/checkpoint/epoch0-iter99"
hf_dir="${model_dir}/convert/epoch0-iter99"
model_name="Mixtral-dMoE_sample"
text="今日はいい天気ですね。"
mp=$(ls -f ${ckpt_dir} | grep .model.pth | wc -l)


echo "##### START #####"
echo "----- merge -----"
python merge_model_weights.py \
	--work_dir ${ckpt_dir} \
	--mp ${mp}


rm -rf ${hf_dir}
mkdir -p ${hf_dir}

echo ""
echo "---- convert -----"
python convert_tokenizer_and_model_to_hugginface.py \
      --input_dir ${ckpt_dir} \
      --output_dir ${hf_dir}


huggingface-cli login --token $HF_TOKEN
huggingface-cli whoami

echo ""
echo "----- upload -----"

python upload_tokenizer_and_model_to_huggingface_hub.py \
	--input_tokenizer_and_model_dir ${hf_dir} \
	--output_model_name ${model_name} \
	--test_prompt_text ${text}

echo ""
echo "##### FINISH #####"
