# Tanuki_pretraining

*注意：  Azure、GCPのVMCにおいて、初期から環境構築を行う為に
        minicondaをインストールします。ネイティブ環境等で、既にminiconda環境がある場合は
        全体に悪影響を及ぼす可能性があります。
        その場合、make_env.sh　の中のコマンドを参考にして、環境を構築してください。


##　gitクローン
```bash
git clone https://github.com/matsuolab/nedo_project_code.git
git checkout team_hatakeyama_phase2 
cd team_hatakeyama_phase2/Tanuki_pretraining/
sudo chmod +x ./make_env.sh 
```


nedo_project_code/team_hatakeyama_phase2/Tanuki_pretraining
    |
    |-- Megatron-LM
    |       |
    |       |-- apex
    |       |
    |       |-- TransformerEngine
    |       |
    |       |-- another
    |
    |
    |-- learning_tokenizer
    |       |
    |       |-- README.md
    |
    |-- tokenize
    |       |
    |       |-- README.md
    |
    |-- training
    |       |
    |       |-- README.md
    |
    |-- make_env.sh

```
## 環境構築





本環境構築シェルスクリプトは、ubuntu 20に対応しています。
ubuntu 22 , ubuntu 24　などは、gcc-11 g++-11を指定してください。
```bash
sudo apt update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
```




$ git clone https://github.com/team-hatakeyama-phase2/Tanuki_pretraining.git
$ cd Tanuki_pretraining
$ make_env.sh
## ---- import OK ----- が表示される
```

$ sudo wget https://github.com/mikefarah/yq/releases/download/v4.30.6/yq_linux_amd64 -O /usr/local/bin/yq
$ sudo chmod +x /usr/local/bin/yq


## トークナイザーの学習
```bash
・learning_tokenizer/train_sentencepiece_tokenizer_mecab.ipynb
## 詳細は　learning_tokenizer　のREADME.md
```

## トークナイズ
```bash
Tanuki_pretraining$ ./tokenize/tokenize.sh　　　##実行場所に注意
## 詳細は　tokenize　のREADME.md
```

## 事前学習
```bash
Tanuki_pretraining$ ./training/one_RTX3090_test.sh    ##実行場所に注意
##詳細は　training　のREADME.md
```

## 謝辞
この成果は、NEDO（国立研究開発法人新エネルギー・産業技術総合開発機構）の助成事業「ポスト５Ｇ情報通信システム基盤強化研究開発事業」（JPNP20017）の結果得られたものです。
