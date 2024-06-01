# README
ライブラリのinstallは、[originalのREADME](./README.md)を参照

```
cd ~/
git clone https://github.com/geniac-team-sannai/ucllm_nedo_prod_MoE.git


cd ~/ucllm_nedo_prod_Moe/train
git clone https://github.com/geniac-team-sannai/Megatron-DeepSpeed-MoE.git
```

## jsonlの作成
サンプルとしてwikipedia_enの一部を取得します

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny`以下にtrain.jsonlが出力されます

```
cd ~/ucllm_nedo_prod_MoE/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny
python download_hf_ds.py
```

※ tokenizerとして"mistralai/Mixtral-8x7B-v0.1"を使う

## jsonl -> binへの変換

```
cd ~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny
sh prepare_dataset.sh
```

※ megatron-coreがライブラリとして入っている場合エラーが起きる?  
原因は、`prepare_dataset.sh`が呼び出している`Megatron-DeepSpeed/tools/preprocess_data.py`がmegatron-coreのbuild_tokenizerを呼び出してしまうため。`Megatron-DeepSpeed/tools/preprocess_data.py`では、`Megatron-DeepSpeed/megatron.tokenizer.build_tokenizer`を呼び出せるよう対応する必要がある。  
とりあえずの対応として、`Megatron-DeepSpeed/tools/preprocess_data.py`の上の方に、`sys.path = [/(HOME)/Megatron-DeepSpeed/megatron] + sys.path`と加えれば動くはず  


### 整形済みjsonl
以下にbin形式として出力済みのデータセットがあるのでこっちを使っても良い

```
/persistentshare/storage/team_sannai/fujimoto/seed_1G

$ ls
dolma_c4_cc_mid_text_document.bin  dolma_c4_cc_mid_text_document.idx
```

dolmaのデータセットのうちc4+common crawlの1Gのtextを変換したもの

mixtralのtokenizerで分割して、`Total tokens: 215.62M`


## 事前学習
```
cd ~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny
```

事前にbrowserからwandbにログインして、projectを作っておく必要がある

コンソールでwandbにログイン

> W&Bにログイン。  
> https://wandb.ai/settings --> Danger Zone --> API keys --> APIキーをコピペ。  
> (.venv) $ wandb login  

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny`にある`ds_config_gpt_TEMPLATE.json`のproject名を適宜修正

```
 "wandb": {
    "enabled": true,
    "project": "sample"
  }
```

※ たぶんここに書けばwandbに連携されるはず。データが飛ぶかは未確認

事前学習の実行


#### gpt2
```
bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--output_model_dir "./output"
```

```
bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--save_interval 200 \
--dataset_index 1 \
--output_model_dir "./output"
```

`./output`以下にlogやtensorboardやcheckpointが出力されます


## 事前学習済みモデル -> HFモデルへの変換

```
cd ~/ucllm_nedo_prod/train/scripts/step3_upload_pretrained_model
```

事前学習時に出力したモデルのディレクトリを引数`input_model_dir`で指定

#### gpt2
```
bash convert_pretrained_model_to_hf.sh \
--output_model_dir ./converted \
--input_model_dir ~/ucllm_nedo_prod_MoE/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny/output/checkpoint/gpt_0.01B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs128_mbs1_g_pp1_seed1234_rebase/global_step1/
```

#### llama2
```
bash convert_pretrained_llama2_to_hf.sh \
--output_model_dir ./converted_llama2 \
--input_model_dir ~/ucllm_nedo_prod_MoE/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny/out_llama2/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs32_g1_pp1_seed1234_rebase/
```

## upload
huggingfaceにloginしておく  
API keyはhuggingfaceのUI上からとってくる  
なければwrite権限で作成  

```
huggingface-cli login
```

huggingfaceのUIからmodel repositoryを作成しておく  

### gpt2

```
python upload_gpt2_to_hf_hub.py \
--repo_id {モデルのrepository_id}  \
--tokenizer_dir mistralai/Mixtral-8x7B-v0.1  \
--test_prompt_text hello  \
--model_dir {HFとして変換したディレクトリを指定}
```


python upload_gpt2_to_hf_hub.py \
--repo_id team-sanai/llama2_tiny \
--tokenizer_dir mistralai/Mixtral-8x7B-v0.1 \
--test_prompt_text hello \
--model_dir ./converted_llama2/


## 以下不明点。調査が必要
### master_addrとmaster_portについて
以下のようなエラーが出ることがある。
```
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
```

おそらく他の人が事前学習を動かしているとportが被る？？？

`~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/gcp_node-1_gpu/wiki-en_gpt_tiny/zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh`では、以下のようにportを変更するようにしてます。

```
DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
```

※ マルチノードのときに使うやつ？変更の影響は不明

### 事前学習で使うtokneinzer
huggingfaceにあるphi-2のtokenizerを使えるように、`--tokenizer-type HFTokenizer`としてます。

```
data_options=" \
    --tokenizer-type HFTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --data-path ${data_path} \
    --data-impl mmap"
```

※ 事前学習のどこでtokenizerを使ってるか調査

### deepspeedのconfigのtemplateファイル
deepspeedのconfigのtemplateファイルを新たに作成してます。

元のpipelineのコードでは、megatron_deepspeedにあるものを使ってるようです。
`template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"`


今回作成したものには以下が追加されています。

```
 "wandb": {
    "enabled": true,
    "project": "sample"
  }
```

