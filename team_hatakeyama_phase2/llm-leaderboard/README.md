# Nejumi-leaderboard Neo for Tanuki
gpt-4o評価対応のNejumi-leaderboard Neoです。

## Set up
1. condaで仮想環境を作成します。
```
# Python仮想環境を作成
conda create -n eval python=3.11 -y

# 作成したPython仮想環境を有効化
conda activate eval

# Python仮想環境を有効化した後は (python3コマンドだけでなく) pythonコマンドも使えることを確認
which python && echo "====" && python --version
```
2. リポジトリをgit cloneし、必要なライブラリをインストールします。
```
# リポジトリをgit clone
git clone https://github.com/team-hatakeyama-phase2/llm-leaderboard.git
cd llm-leaderboard

# 必要なライブラリのインストール
pip install -r requirements.txt
```
3. 以下の環境変数を設定します。
```
export LANG=ja_JP.UTF-8

# https://wandb.ai/settings#api からAPIキーを発行
export WANDB_API_KEY=<your WANDB_API_KEY>

# 共有されたkeyを指定
export OPENAI_API_KEY=<your OPENAI_API_KEY>


# huggingfaceにログイン
huggingface-cli login
```
## Create config.yaml file
configファイルを作成します。configs/config.yamlの設定が評価に使用されます。
1. configテンプレートをコピーします。
```bash
cp configs/alpaca-format-gpt4o.yaml configs/config.yaml
```
2. 以下の項目を編集します。
```# projectとrun_nameの編集
wandb:
  log: True
  entity: "weblab-geniac1" # たぬきチームはweblab-geniac1
  project: "leaderboard_test" # SFT検証の場合leaderboard_sft, テスト用はleaderboard_test
  run_name: 'YOUR_RUN_NAME' # 学習時のproject_name+wandb_nameを記載 e.g. 04_hallucination-tanuki_8B_lora-with_hallucination

# pretrained_model_name_or_pathの編集
model:
  pretrained_model_name_or_path: 'OUTPUT_DIR' #学習したモデルが保存されているpath（or huggingfaceのモデル名）を記載

# pretrained_model_name_or_pathの編集
tokenizer:
  pretrained_model_name_or_path: "OUTPUT_DIR" #学習したモデルが保存されているpathを記載

# basemodel_nameの編集
metainfo:
  basemodel_name: "YOUR_RUN_NAME" # # 学習時のproject_name+wandb_nameを記載 e.g. 04_hallucination-tanuki_8B_lora-with_hallucination

# judge_modelの編集（変更する場合）
mtbench:
  judge_model: 'gpt-4o'

```

   
## Evaluation execution
1. 評価を実行します。

   エラー対策は[こちらのNotionページ](https://www.notion.so/matsuolab-geniac/97d24429af354baa965fc5a5d812601d?pvs=4#c76478bff0e745fbaa260fd13654df8a)をご確認ください。
```
# JMT-Benchのみ動かす場合
python scripts/run_jmtbench_eval.py

# llm-jp-evalのみ動かす場合
python scripts/run_llmjp_eval.py

# llm-jp-eval, JMT-Bench両方動かす場合
python scripts/run_eval.py
```
2. リーダーボードを確認します。
