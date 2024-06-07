# Step 0-2. Python仮想環境の作成
conda create -n accessory python=3.10 -y

mkdir -p ~/miniconda3/envs/accessory/etc/conda/activate.d
echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > ~/miniconda3/envs/accessory/etc/conda/activate.d/edit_environment_variable.sh
echo 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/accessory/lib:$LD_LIBRARY_PATH"' >> ~/miniconda3/envs/accessory/etc/conda/activate.d/edit_environment_variable.sh
chmod +x ~/miniconda3/envs/accessory/etc/conda/activate.d/edit_environment_variable.sh

mkdir -p ~/miniconda3/envs/accessory/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > ~/miniconda3/envs/accessory/etc/conda/deactivate.d/rollback_environment_variable.sh
echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> ~/miniconda3/envs/accessory/etc/conda/deactivate.d/rollback_environment_variable.sh
chmod +x ~/miniconda3/envs/accessory/etc/conda/deactivate.d/rollback_environment_variable.sh
