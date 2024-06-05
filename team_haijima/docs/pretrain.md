# 事前学習の回し方

以下のコマンドを実行することで事前学習を行うことができます。

```bash
cd /storage3
sbatch GENIAC_haijima/train/scripts/jobs/multi_nodes/pretrain_mistral_9b_3node.sh
```

- /storage3の直下で実行する理由は以下の通りです。
  - slurmの出力ファイルが直下に出力されます。
  - wandbのフォルダも実行フォルダ直下に作成されます。

## 学習中の動作確認

- W&Bを参照する
- slurmのログを参照する
  - `tail -f jobs_outs/pretrain_mistral-9b_<jobid>.out`を/storage3の下で実行するとリアルタイムでログを確認できます。

## 障害確認

- 運用監視用Slackに通知が届きます。
- W&Bは学習が動いていても通信が途絶えると通知が届くことがあります。
- 環境にログインして`squeue`で確認してください。（またはW&B）
