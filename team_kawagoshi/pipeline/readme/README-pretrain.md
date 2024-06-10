# Train

## 前提

* 計算環境: g2, 1 node, 1 GPU (Nvidia L4 24GB)
  * 例: `$ srun --partition g2 --nodes=1 --gpus-per-node=1 --time=04:00:00 -c 12 --pty bash -i`

## Step 0. 環境構築

このステップでの目標は、下記のようなディレクトリ構造の状態になることです。

Before:
```sh
~/ucllm_nedo_dev/
└── train/
    ├── scripts/
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

After:
```sh
~/ucllm_nedo_dev/
└── train/
    ├── .venv/
    ├── apex/
    ├── llm-jp-sft/
    ├── Megatron-DeepSpeed/
    ├── scripts/
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

### Step 0-1. Python仮想環境作成前における下準備

```sh
$ cd ~/

# condaのインストール先ディレクトリを作成。
$ mkdir -p ~/miniconda3/ && cd ~/miniconda3/

# condaをインストール。
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh && bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p ~/miniconda3/

# インストールしたcondaを有効化。
$ source ~/miniconda3/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
$ which conda && echo "====" && conda --version
```

### Step 0-2. Python仮想環境の作成

```sh
$ cd ~/ucllm_nedo_dev/train/

# Python仮想環境を作成。
$ conda create --name .venv python=3.9 -y

# Python仮想環境を有効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を編集するように設定。
$ mkdir -p ~/miniconda3/envs/.venv/etc/conda/activate.d
$ echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > ~/miniconda3/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
$ echo 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/.venv/lib:$LD_LIBRARY_PATH"' >> ~/miniconda3/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh
$ chmod +x ~/miniconda3/envs/.venv/etc/conda/activate.d/edit_environment_variable.sh

# Python仮想環境を無効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を元に戻すように設定。
$ mkdir -p ~/miniconda3/envs/.venv/etc/conda/deactivate.d
$ echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > ~/miniconda3/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
$ echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> ~/miniconda3/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh
$ chmod +x ~/miniconda3/envs/.venv/etc/conda/deactivate.d/rollback_environment_variable.sh

# 作成したPython仮想環境を有効化。
$ conda activate .venv

# cuda-11.8.0をインストール。
$ conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# PyTorchを指定のバージョンでインストール。
$ conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Python仮想環境を有効化した後は (python3コマンドだけでなく) pythonコマンドも使えることを確認。
(.venv) $ which python && echo "====" && python --version

# 環境変数 `$PATH` に `$HOME/miniconda3/envs/.venv/bin` が含まれていることを確認。
(.venv) $ echo $PATH

# 環境変数 `$LD_LIBRARY_PATH` に `$HOME/miniconda3/envs/.venv/lib` が含まれていることを確認。
(.venv) $ echo $LD_LIBRARY_PATH
```

### Step 0-3. パッケージ等のインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# PyTorchを指定のバージョンでインストールした後に、requirements.txtを用いて諸々のパッケージをインストール。
(.venv) $ pip install -r ~/ucllm_nedo_dev/train/requirements.txt

# deepspeedの依存パッケージをインストール。
(.venv) $ pip install deepspeed-kernels

# deepspeedを指定のバージョンでインストール。このとき、deepspeed関連の拡張機能たち "ops" を事前にビルドしておくために `DS_BUILD_OPS=1` と設定。
# https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops
# ※しばらく時間がかかるので注意。
(.venv) $ DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.12.4

# deepspeed関連の拡張機能たち "ops" が正しくインストールされていることを確認。
(.venv) $ ds_report
```

### Step 0-4. Megatron-DeepSpeedのインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# Megatron-DeepSpeedのレポジトリをクローン。
(.venv) $ git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/ && git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240205.1.0

# Megatron-DeepSpeedをインストール。
(.venv) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/ && python setup.py install
```

### Step 0-5. apexのインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# apexのレポジトリをクローン。
(.venv) $ git clone https://github.com/NVIDIA/apex

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv) $ cd ~/ucllm_nedo_dev/train/apex/ && git fetch origin && git checkout refs/tags/23.08

# nvccが対応しているCUDAのバージョンとPyTorchが依存しているCUDAのバージョンが一致していることを確認。
(.venv) $ which nvcc && echo "====" && nvcc --version && echo "====" && python -c "import torch; print(torch.version.cuda)"

# pipのバージョンが23.1以上であることを確認。
(.venv) $ which pip && echo "====" && pip --version

# pipのバージョンが23.1以上の場合のインストール方法で、apexをインストール。
# ※しばらく時間がかかるので注意。
(.venv) $ cd ~/ucllm_nedo_dev/train/apex/ && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# apexがインストールされていることを確認。
(.venv) $ pip list | grep "apex"

# apex_C.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
(.venv) $ find ~/ucllm_nedo_dev/train/apex/build/lib.linux-x86_64-cpython-311/ -name apex_C.cpython-311-x86_64-linux-gnu.so
```

### Step 0-6. Flash Attention 2のインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# Flash Attention 2のインストールに必要なninjaを念のため再インストール。
(.venv) $ pip uninstall ninja -y && pip install ninja==1.11.1

# Flash Attention 2をインストール。
(.venv) $ pip install flash-attn==2.5.0 --no-build-isolation

# Flash Attention 2がインストールされていることを確認。
(.venv) $ pip list | grep "flash-attn"
```

### Step 0-7. llm-jp-sftのインストール

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/

# llm-jp-sftのレポジトリをクローン。
(.venv) $ git clone https://github.com/hotsuyuki/llm-jp-sft

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
(.venv) $ cd ~/ucllm_nedo_dev/train/llm-jp-sft/ && git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240208.1.0
```

## Step 1. トークナイザーの学習

### Step 1-1. 学習の実行

Tokenizerはllm-jpのものを使用　./Ucllm_nedo_prod/train/scripts/datasetに保存。

## Step 2. モデルの事前学習

###　事前学習前にすること
# データのshuffle処理が自動実行されるため、https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/gpt_dataset.pyの中で、np_rng.shuffleという処理が３回行われるので、そこをコメントアウトすることで、shuffleを防ぐ。

# また、マルチノード使用時はssh関連を設定する

```jsx
cd
mkdir ~/.ssh
chmod 700 ~/.ssh
touch "${HOME}/.ssh/known_hosts"

#ssh keyの作製
ssh-keygen -t ed25519-sk -C "your_email@example.com"
#自分自身のpubkeyを､authorized_keysにいれる
#pubkeyの表示
cat ~/.ssh/id_ed25519.pub

#pubkeyの内容を反映させる
echo "<公開鍵の内容>" >> ~/.ssh/authorized_keys
```

# bashrcの更新

# .ssh configを作成 (https://github.com/matsuolab/ucllm_nedo_prod/blob/main/train/scripts/common/create_ssh_config_file_for_gcp_play_multi_node_multi_gpu.shのスクリプトをコピペしたものです)
#以下shファイルを作って実行する(node条件を変えたら再実行する必要あり)

# 親ノードが各ノードにパスフレーズなしでSSHアクセスできるように設定した~/.ssh/configファイルを作成。
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)


ssh_config_file="${HOME}/.ssh/config"
echo "" > "${ssh_config_file}" 

for node in $nodes; do
    # Update the known_hosts file for each node, removing old keys
    ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "$node"
    # Add new node configuration to the SSH configuration file
    echo "Host $node" >> "${ssh_config_file}"
    echo "    HostName $node" >> "${ssh_config_file}"
    echo "    Port 22" >> "${ssh_config_file}"
    echo "    StrictHostKeyChecking no" >> "${ssh_config_file}"
    echo "" >> "${ssh_config_file}"
done

echo "SSH configuration has been updated."
cat ${ssh_config_file}

# また、pretrainファイルのcondaパス(282行目)を自分のcondaに書き換える。

### Step 2-1. 事前学習の実行

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step2_pretrain_model/

# W&Bにログイン。
# https://wandb.ai/settings --> Danger Zone --> API keys --> APIキーをコピペ。
(.venv) $ wandb login

# W&Bにログインしていることを確認。
(.venv) $ cat ~/.netrc

# 事前学習済みのデータを次に保存済み /persistentshare/storage/team_kawagoshi/nishijima/datasets/small_data_text_document
  *RAMが潤沢な場合は次でも可　/persistentshare/storage/team_kawagoshi/nishijima/datasets/large_data_text_document.bin

# 事前学習スクリプトを実行。
(.venv) $ bash ~/ucllm_nedo_prod/train/scripts/step2_pretrain_model
(.venv) $ bash ./pre_dev/pretrain_llama2_1node.sh \
    --output_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/${YourName}/llama2-0.3B \
    --save_interval 200

# Mistral 系列の場合は以下を事前に実行
# Megatron-deepspeedのattention関連のファイルを書き換え。
(.venv) $ bash ./pre_dev/setup.sh  \
```

### Step 2. でのトラブルシューティング

##### 1. "ImportError: cannot import name 'helpers' from 'megatron.data' (Megatron-DeepSpeed/megatron/data/__init__.py)" というエラーが出た場合

原因: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/helpers.cpython-311-x86_64-linux-gnu.so` が正しく作成されていないことが原因と考えられます。

解決策: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefile` 内に記載されている `python3-config` のパスを `$ which python3-config` で出力された絶対パスに変更してから、 `~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/` にて `make` コマンドを実行してみて下さい。

```sh
# python3-configの絶対パスを確認。
(.venv) $ which python3-config

# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefileのpython3-configのパスを、上記のwhichコマンドで出力された絶対パスに変更。
# 上記作業は不要。逆にfp16をdisableにする。
(.venv) $ vim ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/Makefile 
"""
# Before
LIBEXT = $(shell python3-config --extension-suffix)

# After
LIBEXT = $(shell /absolute/path/to/python3-config --extension-suffix)
"""

# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/にてmakeコマンドを実行。
(.venv) $ cd ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/ && make

# helpers.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
(.venv) $ find ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/data/ -name helpers.cpython-311-x86_64-linux-gnu.so
```

参考リンク: <br/>
* https://zenn.dev/turing_motors/articles/04c1328bf6095a#pyenv-virtualenv-%E3%82%92%E4%BD%BF%E3%81%86%E3%81%A8%E5%BF%85%E8%A6%81%E3%81%AB%E3%81%AA%E3%82%8B%E5%87%A6%E7%90%86
* https://zenn.dev/turing_motors/articles/da7fa101ecb9a1#makefile%E3%81%AE%E6%9B%B8%E3%81%8D%E6%8F%9B%E3%81%88

#### 2. 事前学習スクリプトが "> compiling and loading fused kernels ..." というところでスタックした場合

原因: <br/>
既存の `~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/` が作成された当時と現在でハードウェアやCUDAのバージョンが異なっていることが原因と考えられます。

解決策: <br/>
`~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/` を削除してから、もう一度事前学習スクリプトを実行してみて下さい。

```sh
# ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/を削除。
(.venv) $ rm -rf ~/ucllm_nedo_dev/train/Megatron-DeepSpeed/megatron/fused_kernels/build/
```

参考リンク: <br/>
* https://github.com/NVIDIA/Megatron-LM/issues/82#issuecomment-1613749424

## Step 3. 事前学習済みモデルのアップロード

### Step 3-1. トークナイザーと事前学習済みモデルのHuggingFace Transformers形式への変換


```sh
#事前にファイルを以下にコピーする。
from common (deepspeed_checkpoint.py, deepspeed_to_megatron.py, ds_to_unversal.py) to Megatron-DeepSpeed/tools/convert_checkpoint
#以下は必要に応じてコピー。
from common universal_checkpoint.py to deepspeed/checkpoint

# TPが2以上の場合は次を実行。これでuniversal checkpointが作成される。zero stage2の場合は下に記載。
(.venv) $ cd ~
(.venv) $ bash python ucllm_nedo_dev/train/Megatron-DeepSpeed/tools/convert_checkpoint/ds_to_universal.py   \
  --input_folder ../../persistentshare/storage/team_kawagoshi/nishijima/test_change_tpz/llama2-0.3B_tp1_pp1_zero1_v2/checkpoint/global_step_2200   \
  --output_folder ../../persistentshare/storage/team_kawagoshi/nishijima/test_change_tpz/llama2-0.3B_tp1_pp1_zero1_v2/checkpoint/global_step2200_universal

# TPが2以上の場合は次にtrainを通してlayer層を作成。実はこれは不要の可能性も。実行前にはskip_train=Trueに設定かつelseとして以下を記載。
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)

(.venv) $　cd ~
(.venv) $　cd ucllm_nedo_dev/train/scripts/step2_pretrain_model

(.venv) $　bash ./convert_llama2.sh   \
  --input_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/nishijima/test_change_tpz/llama2-0.3B_tp1_pp1_zero1_v2 \
  --output_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/nishijima/changed_tpz/llama2-0.3B/111_to_220

# zero stage2の場合は次に変換。ただし、TP>1では未確認
ツリー構造を
<pre>
model_name
└── checkpoint
        └── global_stepxxx
                └─ model_optim_rng.pt
から
model_name
└── checkpoint
        ├── iter_xxxxxxx
        ├        └─ mp_rank_00
        ├               └─ model_optim_rng.pt
        └── latest_checkpoint_iteration.txt

</pre>

# 変換スクリプトを実行。
(.venv) $　cd ~
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/
(.venv) $ bash ./convert_toke_and_model_to_mega_to_hf.sh \
    --input_tokenizer_file ~/ucllm_nedo_dev/train/dataset/code10k_en20k_ja30k.ver2.1.model \
    --input_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/${YourName}/llama2-0.3B/checkpoint/${Your_JobName}/ \
    --output_tokenizer_and_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/${YourName}/llama2-0.3B/checkpoint/${Your_JobName}/step3_upload_pretrained_model/ \
    --temp_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/${YourName}/llama2-0.3B/checkpoint/${Your_JobName}/temp/ \
    --model_name Llama2 or Mistral
```

### Step 3-2. トークナイザーと事前学習済みモデルのHuggingFace Hubへのアップロード

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

# HuggingFaceにログイン。
# https://huggingface.co/settings/tokens --> 書き込み権限ありのAPIキーをコピペ。
(.venv) $ huggingface-cli login

# HuggingFaceにログインしていることを確認。
(.venv) $ huggingface-cli whoami


# Megatron-deepspeedのファイルを書き換え。
(.venv) $ bash ./setup.sh \

# アップロードスクリプトを実行。
(.venv) $ python ./upload_tokenizer_and_pretrained_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir ../../../../../../persistentshare/storage/team_kawagoshi/${YourName}/llama2-0.3B/checkpoint/${Your_JobName}/temp/ \
    --output_model_name gpt_0.125B_global_step1000 \
    --test_prompt_text "Once upon a time,"
```

## Step 4. モデルのファインチューニング

### Step 4-1. ファインチューニングの実行
#使用するデータセットの追加が必要。


```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step4_finetune_model/

# ファインチューニングスクリプトを実行。 (HuggingFaceにアップロードした事前学習モデルをダウンロードして使用する場合)
(.venv) $ bash ./pre_dev/train_ja_1node.sh --input_model_name_or_path ${YOUR_HUGGINGFACE_USEandMODELRNAME} \
    --output_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/

# ファインチューニングスクリプトを実行。 (ローカルに保存してある事前学習モデルをそのまま使用する場合)
(.venv) $ bash ./pre_dev/train_ja_1node.sh --input_model_name_or_path ~/ucllm_nedo_dev/train/output/step3_upload_pretrained_model/gpt_0.125B_global_step1000/ \
    --output_dir ~/ucllm_nedo_dev/train/output/step4_finetune_model/gpt_0.125B_global_step1000_openassistant/
```

## Step 5. ファインチューニング済みモデルのアップロード

### Step 5-1. トークナイザーとファインチューニング済みモデルのHuggingFace Hubへのアップロード

```sh
(.venv) $ cd ~/ucllm_nedo_dev/train/scripts/step5_upload_finetuned_model/

# HuggingFaceにログインしていることを確認。
(.venv) $ huggingface-cli whoami

# アップロードスクリプトを実行。
(.venv) $ python ./upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir ~/ucllm_nedo_dev/train/output/ \
    --output_model_name gpt_0.125B_global_step1000_openassistant \
    --test_prompt_text "<s>[INST] <<SYS>>\nあなたは日本語で返答する優秀なAIアシスタントです。\n<</SYS>>\n\n apple watchでできることリストを教えてください。[/INST]"
```
