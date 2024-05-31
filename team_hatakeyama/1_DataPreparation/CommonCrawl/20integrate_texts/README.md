# データセットの統合

## データを決めます｡
- どのデータを用いるかについては､[dataset_dict](./dataset_dict.py)を直接いじって作業します｡
    - output_pathを指定すること
    - [日英のデータセット](team_hatakeyama/1_DataPreparation/20integrate_texts/dataset_dict.py)
        - メインの学習に使用。
        - tokenize時は、scale=200にした小型データを使用
    - [日本語のみのデータセット](team_hatakeyama/1_DataPreparation/20integrate_texts/real_btm/BTM_ja_dataset_dict.py)
        - 2 epoch目に使用

- [notebook](./01check_distribution.ipynb)で分布なども確認出来ます｡

## ここでは､HuggingFaceのDatasetsライブラリなどを諸々loadして､一つのjsonlを書き出します
- [loader](./src/loaders.py)を定義しておきます｡
- 用いるDatasetは､dataset_dictに記入していきます｡

## BTM用に､データをクラスタリングし直します 
- [リクラスタリング](./00reclustering.ipynb)
    - 1万クラスタ　to 5 クラスタするためのモデルの生成
- [各クラスタのサイズ確認](./01CountClusterSizes.py)
- [クラスタリングされたデータセットの分布確認](./02check_distribution.ipynb)
- [データセットの統合](./03integrate_dataset.py)


