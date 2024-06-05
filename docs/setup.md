# 個人別セットアップ

## はじめに

- このドキュメントは、各個人の本番環境を構築するための手順を記載しています。
- gcloudまでのログインについては、各自で行ってください。
  - https://docs.google.com/document/d/1IgQ5gXXvtfxGgEWMG1spxZD-f3levamAkDkp2qtZKAc
- **特に4/26 14:00以降に新規ログインしたユーザは共有ディスクへのアクセス権がないためSlackにて連絡が必要になります。**
- 以下の作業はログイン後、計算ノード上で行ってください。
```
srun --partition a3 --nodes=1 --gpus-per-node=0 --nodelist=slurm0-a3-ghpc-[6-8] --pty bash -i
```
- なお、事前にminicondaのインストールが終了していることを前提としています。
  - minicondaのインストールは誰か一人が行えば良く、各自が行う必要はありません。

## ~/.bashrcの編集

- 本番環境には/opt/condaがインストールされており、ログイン時に有効化されていることがあります。
  - 運営側に自動更新するのを辞めて頂いたので設定されていない場合もあります。
  - 設定されていなければ、この作業は不要です。
- sbatchでは/storage3/miniconda3を使うように指定をしていますので、設定がある場合には`~/.bashrc`の最後にあるconda initialize以下の部分をコメント化してください。
- **.bashrcのcondaの初期化が残っていると、/opt/condaを使い続けてしまうようなので、必ずコメント化してください**

## W&Bへのログイン

- 以下のコマンドを実行して、wandbにログインしてください。

```bash
source /storage3/miniconda3/etc/profile.d/conda.sh
conda activate .ds0_14_2
wandb login
```

## deepspeedの設定ファイル

- deepspeedの設定を行うためのファイルを用意してください。
- `~/.deepspeed_env`ファイルを作成してください。
  - このファイルは共通化できそうなので、システム的に定義する可能性もあります。
- 以下の内容を記述してください。(NCCLの環境設定は、今後変更される可能性がありますので、Slackやこのドキュメントを注視してください。)

```ini
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=enp0s12
PATH=/storage3/miniconda3/envs/.ds0_14_2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/snap/bin
```
