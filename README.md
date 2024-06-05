# GENIAC_haijima

Team 甲（きのえ）の提出版モデルの作成に用いたコードを以下に示します。

※ 元となるソースコードには、Microsoft社の[Megatron-Deepspped](https://github.com/microsoft/Megatron-DeepSpeed)、ABEJA社の[コード](https://github.com/abeja-inc/Megatron-LM)、東大松尾研の[標準コード](https://github.com/matsuolab/ucllm_nedo_prod)が含まれます。





## 環境整備
[コチラ](https://github.com/GENIAC-team-haijima/GENIAC_haijima/blob/main/docs/condaenv.md)のドキュメントをご参照ください。





## トークナイザ
[コチラ](https://github.com/GENIAC-team-haijima/GENIAC_haijima/tree/main/train/scripts/step1_train_tokenizer)のコードを用いました。





## 事前学習
```
GENIAC_haijima/train/scripts/jobs/multi_nodes/pretrain_mistral_9b_3node.sh
```

※ 事前学習の回し方は[コチラ](https://github.com/GENIAC-team-haijima/GENIAC_haijima/blob/main/docs/pretrain.md)をご参照ください。





## 事後学習

事後学習用のコードは[コチラ](https://www.notion.so/matsuolab-geniac/LoRA-e0074a53061a49e28a6f71e5b9cc7def#83a9fc15ebca4240a4fef7dc877d865c)をご参照ください。




