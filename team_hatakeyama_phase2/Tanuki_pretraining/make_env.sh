#!/bin/bash

DIR=`readlink -f $PWD`

tev=1.7
fav=2.5.7
python_Mver="3"
python_mver="10"
conda_ver="py${python_Mver}${python_mver}_23.10.0-1"
env_name="tanuki_pretraining_py${python_Mver}${python_mver}_TEv${tev}_FAv${fav}"

echo "----------------------------------------"
echo "##### START : ${env_name} #####"
echo "----------------------------------------"
echo ""

echo "########## git clone : requirementsの変更 ##########"
sed -i \
	-e "/^mpi4py/s/^/# /g" \
	-e "s/^numpy$/numpy==1.26.4/g" \
	$DIR/Megatron-LM/requirements.txt
# python3-config：パス修正（llmjp_tsubameのユーザパス残存のため）
sed -i 's@/gs/fs/tga-bayes-crest/taishi/pyenv/versions/3.10.14/bin/@@'  $DIR/Megatron-LM/megatron/core/datasets/Makefile
# wandb：マルチノードログ対応
sed -i \
	-e "/if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):/i \    local_last_rank = args.rank%torch.cuda.device_count() - (torch.cuda.device_count() -1)" \
	-e "s/if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):/if getattr(args, 'wandb_project', '') and local_last_rank==0:/" \
	-e 's/exp_name = args.wandb_name + "-" + now/exp_name = args.wandb_name + "-" + now + "_" + str(args.rank)/' \
	$DIR/Megatron-LM/megatron/training/global_vars.py


# echo "########## git clone : Megatron-LM(llmjp), apex, Transformer Engine(${tev}) ##########"
# if [[ ! -d $DIR/Megatron-LM ]]; then
	# git clone -b tsubame https://github.com/llm-jp/Megatron-LM.git
	# # mpi4py：condaでインストールするためコメントアウト
	# # numpy : v1.26.0をインストール
	# sed -i \
	# 	-e "/^mpi4py/s/^/# /g" \
	# 	-e "s/^numpy$/numpy==1.26.4/g" \
	# 	$DIR/Megatron-LM/requirements.txt
	# # python3-config：パス修正（llmjp_tsubameのユーザパス残存のため）
	# sed -i 's@/gs/fs/tga-bayes-crest/taishi/pyenv/versions/3.10.14/bin/@@'  $DIR/Megatron-LM/megatron/core/datasets/Makefile
	# # wandb：マルチノードログ対応
	# sed -i \
	# 	-e "/if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):/i \    local_last_rank = args.rank%torch.cuda.device_count() - (torch.cuda.device_count() -1)" \
	# 	-e "s/if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):/if getattr(args, 'wandb_project', '') and local_last_rank==0:/" \
	# 	-e 's/exp_name = args.wandb_name + "-" + now/exp_name = args.wandb_name + "-" + now + "_" + str(args.rank)/' \
	# 	$DIR/Megatron-LM/megatron/training/global_vars.py
	
	# cd $DIR/Megatron-LM && git clone https://github.com/NVIDIA/apex.git

	# cd $DIR/Megatron-LM && git clone --branch release_v${tev} --recursive https://github.com/NVIDIA/TransformerEngine.git
# fi

echo "########## install : miniconda3(${conda_ver}) ##########"
echo "Deleting $DIR/miniconda3..."
rm -rf "$DIR/miniconda3"

if [[ ! -d $DIR/miniconda3 ]]; then
	mkdir -p $DIR/miniconda3/ && cd $DIR/miniconda3/
	wget https://repo.anaconda.com/miniconda/Miniconda3-${conda_ver}-Linux-x86_64.sh && bash Miniconda3-${conda_ver}-Linux-x86_64.sh -b -u -p $DIR/miniconda3
fi

start_time=`date +%s`

source $DIR/miniconda3/etc/profile.d/conda.sh

echo "########## conda create : python ##########"
conda create -p $DIR/miniconda3/envs/$env_name python=${python_Mver}.${python_mver} -y

mkdir -p $DIR/miniconda3/envs/$env_name/etc/conda/activate.d && \
echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $DIR/miniconda3/envs/$env_name/etc/conda/activate.d/edit_environment_variable.sh && \
echo 'export LD_LIBRARY_PATH="'$DIR'/miniconda3/envs/'$env_name'/lib:$LD_LIBRARY_PATH"' >> $DIR/miniconda3/envs/$env_name/etc/conda/activate.d/edit_environment_variable.sh && \
echo 'export CUDNN_PATH="'$DIR'/miniconda3/envs/'$env_name'/lib/python'${python_Mver}.${python_mver}'/site-packages/nvidia/cudnn"' >> $DIR/miniconda3/envs/$env_name/etc/conda/activate.d/edit_environment_variable.sh && \
chmod +x $DIR/miniconda3/envs/$env_name/etc/conda/activate.d/edit_environment_variable.sh

mkdir -p $DIR/miniconda3/envs/$env_name/etc/conda/deactivate.d && \
echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > $DIR/miniconda3/envs/$env_name/etc/conda/deactivate.d/rollback_environment_variable.sh && \
echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> $DIR/miniconda3/envs/$env_name/etc/conda/deactivate.d/rollback_environment_variable.sh && \
echo 'unset CUDNN_PATH' >> $DIR/miniconda3/envs/$env_name/etc/conda/deactivate.d/rollback_environment_variable.sh && \
chmod +x $DIR/miniconda3/envs/$env_name/etc/conda/deactivate.d/rollback_environment_variable.sh


echo "########## conda activate ##########"
conda activate $env_name
echo "ORIGINAL_LD_LIBRARY_PATH	-> ${ORIGINAL_LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH		-> ${LD_LIBRARY_PATH}"
echo "PATH			-> ${PATH}"
echo "CUDNN_PATH		-> ${CUDNN_PATH}"


echo "########## conda install : cuda-toolkit, cudnn, nccl, mpi4py ##########"
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y
conda install -c conda-forge cudnn=8.9.7 nccl=2.18.3 mpi4py=3.1.6 -y
conda install -c conda-forge cmake -y

echo "########## pip install : pip & requirements ##########"
pip install pip==24.0
cd $DIR/Megatron-LM && pip install -r requirements.txt

echo "########## pip install : apex ##########"
cd $DIR/Megatron-LM/apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

echo "########## pip install : Transfomer Engine ##########"
cd $DIR/Megatron-LM/TransformerEngine && pip install .

echo "########## pip install : Flash Attention ##########"
cd $DIR && pip install --no-build-isolation flash-attn==$fav

echo "########## conda install for tokenizer ##########"
# conda install scikit-learn=1.5.1 nltk=3.8.1 sentencepiece=0.2.0 pytz=2024.2 pandas=2.2.2 tqdm=4.66.5 transformers=4.42.4 -y

echo "===== import check : transformer_engin & flash_attn ====="
python -c "from transformer_engine import pytorch" && echo "----- import OK -----" || echo "----- import NG -----"

end_time=`date +%s`

run_time=$((end_time - start_time))

echo ""
echo "----------------------------------------"
echo "##### FINISH : ${run_time}s #####"
echo "----------------------------------------"

cd $DIR && conda deactivate
