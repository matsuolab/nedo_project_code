# フォルダ構成

- 実行環境としてのフォルダ構成を以下のようにする。
- なお、本番環境におけるGENIAC_ROOT_DIRは`/storage3`となる。

```
${GENIAC_ROOT_DIR}
+ dataset_pre
  + 20240418_preprocess
    + tokenizer                # merged.model、merged.vocabが格納される。
    + datasets
      + merged_japanese        # japanese.jsonl 事前学習用データセット
+ train_results                # 学習結果(TODO: 事後学習ファイルも用意する)
  + pretrained_models
    + checkpoint
    + deepspeed_config
    + log
    + tensorboard
+ jobs_outs                    # SBATCHジョブの出力
+ wandb                        # W&Bのログ
+ GENIAC_haijima               # リポジトリ一式
  + train
    + Megatron_DeepSpeed       # リポジトリ内のMegatron-DeepSpeed
    + scripts
      + jobs                   # SBATCH実行用スクリプト
      + step1_train_tokenizer  # 
      + step2_pretrain_model   # 
      + ...
```
