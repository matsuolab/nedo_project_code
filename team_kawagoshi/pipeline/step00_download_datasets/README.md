## 環境構築

### Pythonのバージョン確認

Python>=3.11 がインストールされているものとする

確認方法は以下
```sh
$ python --version
// Python 3.11.7
```

もしPythonがインストールされていない場合は[Python.jp](https://www.python.jp/install/centos/index.html)を参考にインストールする

### 必要なライブラリのダウンロード

step00_download_datasetsディレクトリにいることを確認した上でダウンロードを行なってください

```sh
$ cd ~/nedo_project_code/team_kawagoshi/pipeline/step00_download_datasets/download_datasets
$ pwd
// ~/nedo_project_code/team_kawagoshi/pipeline/step00_download_datasets/download_datasets 
```

## 1. Download datasets

ストリーミングでダウンロードを行う場合
```sh
$ python main.py --dataset=mc4-ja --split=train --streaming=True --index_from=0 --index_to=10
```

全日本語mC4をダウンロードする場合
```sh
$ python main.py --dataset=mc4-ja --split=train --streaming=False --index_from=0 --index_to=10
```

git-lfsにて、高速ダウンロードする場合
```sh
$ cd ~/nedo_project_code/team_kawagoshi/pipeline/step00_download_datasets/output 
$ git lfs install
$ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/izumi-lab/mc4-ja
$ git lfs fetch
$ cd mc4-ja
$ git lfs pull
```