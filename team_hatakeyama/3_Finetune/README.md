# Finetuning 
- 事前学習済みモデルのファインチューニングを行う

# Setup
## ファインチューニングのライブラリ群
~~~
conda create -n llmeval python=3.11 -y
conda activate llmeval
#llm-jp-sftのレポジトリに移動
cd llm-jp-sft
export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/llmeval/bin:$PATH
pip install -r requirements.in 
pip install flash-attn --no-build-isolation #flash atten
pip install --upgrade accelerate #accelerateを最新版に
pip install datasets

#evalのリポジトリ
cd 4_eval/llm-leaderboard
conda create -n llmeval python=3.11 -y
conda activate llmeval
pip3 install -r llm-leaderboard/requirements.txt

~~~


## wandb関連
- 評価用に､[新規プロジェクトをwandb上で作ります](https://wandb.ai/new-project)
- [config](./4_eval/llm-leaderboard/configs/config_eval.yaml)を修正します｡
  - entity,projectは､上記のurlから作成したもの
  - run_nameは適当
~~~
entity: "your_org_name" 
project: "llmeval" 
run_name: "test1" 
~~~


## 実行
### ファインチューニング
- pyファイルを実行する
~~~
0524ft_run.py data/0524clean_halcination_little_codes 1_0524clean_halcination_little_codes

~~~
# チームで作ったモデル
- モデルA
  - [ハルシネーションをルールベースで削除したデータセットの生成](./ft/0524_1_dataset_clean_halcination.py)
  - 3 epochを学習
- モデルB
  - [ハルシネーションを許容したデータセットの作成](./ft/0524_2_dataset_with_halcination.py) 
  - 3 epochを学習

- 最後に、[mergekit_moe](./X_merge/kit.ipynb)でMoEマージを実施
