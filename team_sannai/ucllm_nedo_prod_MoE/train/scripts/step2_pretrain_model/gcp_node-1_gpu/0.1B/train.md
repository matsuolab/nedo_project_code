# 事前学習用README

```
cd ~/ucllm_nedo_prod_MoE/train/scripts/step2_pretrain_model/gcp_node-1_gpu/0.1B
```

## 準備
`output_model_dir`については適宜使いやすいものに変えても大丈夫です

weight and biasへのログインがまだであれば、cliからログイン
```
$ wandb login
```

`ds_config_gpt_TEMPLATE.json`の
```
  "wandb": {
    "enabled": true,
    "project": "sample"
  }
```
は作成したproject名に変更


### set1
`/persistentshare/storage/team_sannai/team_la/seed/dataset1`のデータセットを使います

```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--lr_warmup_steps 10000 \
--dataset_index 1 \
--output_model_dir "./gpt_seed"
```


### set2
`/persistentshare/storage/team_sannai/team_la/seed/dataset2`のデータセットを使います

```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--lr_warmup_steps 5000 \
--dataset_index 2 \
--output_model_dir "./gpt_seed"
```


### set3
`/persistentshare/storage/team_sannai/team_la/seed/dataset3`のデータセットを使います

```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--lr_warmup_steps 1000 \
--dataset_index 3 \
--output_model_dir "./gpt_seed"
```


## error
```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:6000 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
```

のようなエラーが出た場合は、


```
MASTER_ADDR=localhost
MASTER_PORT=6000

DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
```

の`MASTER_PORT`を別のものに変更してください

## memo
一旦warmupのみ変更

```
lr=6.0e-4
min_lr=1.0e-6
batch_size=32
dataset_size=3B
```

学習率についてはpipelineのサンプル通り  
https://github.com/geniac-team-sannai/ucllm_nedo_prod_MoE/blob/dev/train/scripts/s[…]e-1_gpu/wiki-en_gpt_tiny/zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh

