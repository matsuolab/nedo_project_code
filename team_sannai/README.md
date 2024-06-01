# team-sannai code
## 構成
- tokenizer
- ucllm_nedo_prod_MoE
- mergoo

### tokenizer
tokenizerの学習/評価用のscriptです

### ucllm_nedo_prod_MoE
pretrainおよびexpert学習用のコード群です

以下のrepositoryのforkです  
https://github.com/matsuolab/ucllm_nedo_prod

以下に事前学習用のコードが追加されています

`ucllm_nedo_prod_MoE/train/scripts/step2_pretrain_model/gcp_node-n_gpu-n/10B`

以下にexpert学習用のloraでのFTコードが追加されています

`ucllm_nedo_prod_MoE/train/scripts/step2-1_lora_ft`


`ucllm_nedo_prod_MoE/train`以下に`Megatron-DeepSpeed-MoE/`が追加してあります  

`Megatron-DeepSpeed-MoE`は以下のforkです
https://github.com/microsoft/Megatron-DeepSpeed


## mergoo
loraで学習したexpertをマージ/評価を行うためのコード群です

以下のrepositoryのforkです  
https://github.com/Leeroo-AI/mergoo

