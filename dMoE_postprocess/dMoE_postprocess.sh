#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=slurm0-a3-ghpc-[18-20]
#SBATCH --job-name=postprocess
#SBATCH --output=%x_%j.out
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=24


. ~/miniconda3/etc/profile.d/conda.sh
conda activate accessory

#
###### manual setting #####
#
training_type="pretrain"
data_type="pretrain_0520"
epoch=0
iter=134999

#
###### automatic setting #####
#
#ckpt_dir="/storage7/${training_type}/Mixtral_3node_${data_type}/epoch${epoch}"
ckpt_dir="/storage7/${training_type}/Mixtral_3node_${data_type}/epoch${epoch}-iter${iter}"
#model_name="dMoE_8B_${data_type}_epoch${epoch}"
model_name="dMoE_8B_${data_type}_iter${iter}_no_rope"
test_prompt="今日の天気は"
merge_dir="${ckpt_dir}/merge_no_rope"
hf_convert_dir="${ckpt_dir}/convert_no_rope"
mps=$(ls -f ${ckpt_dir} | grep .model.pth | wc -l)
eps=$(grep '"num_experts"' ${ckpt_dir}/config.json | cut -d ":" -f 2 | tr -d ",")
n_layers=$(grep '"n_layers"' ${ckpt_dir}/config.json | cut -d ":" -f 2 | tr -d ",")
n_heads=$(grep '"n_heads"' ${ckpt_dir}/config.json | cut -d ":" -f 2 | tr -d ",")
n_kv_heads=$(grep '"n_kv_heads"' ${ckpt_dir}/config.json | cut -d ":" -f 2 | tr -d ",")

#
###### postprocess #####
#
begin_time=`date +%s`
echo "##### START $(TZ=Asia/Tokyo date +%F_%T) #####"
#===========================================
rm -rf ${merge_dir}
mkdir -p ${merge_dir}
python merge/merge_model_weights.py \
	--input_dir ${ckpt_dir} \
	--output_dir ${merge_dir} \
	--mps ${mps} \
	--eps ${eps}
merge_time=`date +%s`
echo "--------------------"
echo "OK : merge $((${merge_time} - ${begin_time}))s"
sstat -j ${SLURM_JOB_ID}.batch -o MaxVMSize,MaxRSS
echo "--------------------"
echo ""
#-------------------------------------------
python merge/check_merge.py ${ckpt_dir} ${merge_dir} ${mps} ${eps}
merge_check_time=`date +%s`
echo "--------------------"
echo "OK : merge check $((${merge_check_time} - ${merge_time}))s"
sstat -j ${SLURM_JOB_ID}.batch -o MaxVMSize,MaxRSS
echo "--------------------"
echo ""
#===========================================
rm -rf ${hf_convert_dir}
mkdir -p ${hf_convert_dir}
python convert/convert_tokenizer_and_model_to_hugginface.py \
	--input_dir ${merge_dir} \
	--output_dir ${hf_convert_dir}
convert_time=`date +%s`
echo "--------------------"
echo "OK : convert $((${convert_time} - ${merge_check_time}))s"
sstat -j ${SLURM_JOB_ID}.batch -o MaxVMSize,MaxRSS
echo "--------------------"
echo ""
#-------------------------------------------
python convert/check_convert.py ${merge_dir} ${hf_convert_dir} ${eps} ${n_layers} ${n_heads} ${n_kv_heads}
convert_check_time=`date +%s`
echo "--------------------"
echo "OK : convert check $((${convert_check_time} - ${convert_time}))s"
sstat -j ${SLURM_JOB_ID}.batch -o MaxVMSize,MaxRSS
echo "--------------------"
echo ""
#===========================================
huggingface-cli login --token $HF_TOKEN
huggingface-cli whoami
#-------------------------------------------
python upload/upload_tokenizer_and_model_to_huggingface_hub.py \
	--input_tokenizer_and_model_dir ${hf_convert_dir} \
	--output_model_name ${model_name} \
	--test_prompt_text ${test_prompt} \
	--use_orgs \
	#--no_upload
upload_time=`date +%s`
echo "--------------------"
echo "OK : upload  $((${upload_time} - ${convert_check_time}))s"
sstat -j ${SLURM_JOB_ID}.batch -o MaxVMSize,MaxRSS
echo "--------------------"
echo ""
#===========================================
echo "##### FINISH  $(TZ=Asia/Tokyo date +%F_%T) #####"
