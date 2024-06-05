#/bin/bash
set -e

GENIAC_ROOT_DIR=/storage3
CONDA_DIR=$GENIAC_ROOT_DIR/miniconda3
GENIAC_SOURCE_DIR=$GENIAC_ROOT_DIR/GENIAC_haijima

mkdir -p $CONDA_DIR
cd $CONDA_DIR

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh
bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p $CONDA_DIR

source $CONDA_DIR/etc/profile.d/conda.sh
which conda && echo "====" && conda --version

conda deactivate
conda create --name .venv python=3.11 -y

mkdir -p $CONDA_DIR/envs/.venv/etc/conda/activate.d
echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $CONDA_DIR/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
echo "export GENIAC_ROOT_DIR=${GENIAC_ROOT_DIR}" >> $CONDA_DIR/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
echo "export CONDA_DIR=${CONDA_DIR}" >> $CONDA_DIR/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
echo 'export LD_LIBRARY_PATH="$CONDA_DIR/envs/.venv/lib:$LD_LIBRARY_PATH"' >> $CONDA_DIR/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
chmod +x $CONDA_DIR/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh

mkdir -p $CONDA_DIR/envs/.venv/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > $CONDA_DIR/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> $CONDA_DIR/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
echo 'unset GENIAC_ROOT_DIR' >> $CONDA_DIR/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
echo 'unset CONDA_DIR' >> $CONDA_DIR/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
chmod +x $CONDA_DIR/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh

conda activate .venv
which python && echo "====" && python --version

echo $PATH
echo $LD_LIBRARY_PATH

conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install pip==24.0
pip install -r $GENIAC_SOURCE_DIR/requirements.txt
pip install deepspeed-kernels
DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.12.4

cd $GENIAC_SOURCE_DIR/train/Megatron-DeepSpeed
python setup.py install

cd $GENIAC_SOURCE_DIR/train
git clone https://github.com/NVIDIA/apex.git
cd apex
git fetch origin
git checkout refs/tags/23.08

which nvcc && echo "====" && nvcc --version && echo "====" && python -c "import torch; print(f'{torch.version.cuda = }')"
which pip && echo "====" && pip --version
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
pip list | grep "apex"
find $GENIAC_SOURCE_DIR/train/apex/build/lib.linux-x86_64-cpython-311/ -name apex_C.cpython-311-x86_64-linux-gnu.so

cd $GENIAC_SOURCE_DIR/train
pip uninstall ninja -y && pip install ninja==1.11.1
pip install flash-attn==2.5.0 --no-build-isolation
pip list | grep "flash-attn"

cd $GENIAC_SOURCE_DIR/train
git clone https://github.com/hotsuyuki/llm-jp-sft.git
cd llm-jp-sft
git fetch origin
git checkout refs/tags/ucllm_nedo_dev_v20240415.1.0
