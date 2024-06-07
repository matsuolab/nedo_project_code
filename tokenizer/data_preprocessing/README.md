# データ前処理
トークナイザー学習のために、テキストは.txtファイルとして作成する。

## 日本語テキストの前処理
- 日本語以外のテキストは前処理をせずそのまま用いる。

## 形態素解析
### fugashiをインストール
```shell
pip install fugashi[unidic]
python -m unidic download
```
### スクリプトの実行
- データは適当な格納先に置いておく。
```shell
python -m keitaisokaiseki \
--input dataset/sample.txt\
--output dataset/wakachi/sample_wakachi.txt
```
### 出力サンプル
```shell
アバター||||を||||自分||||だ||||と||||思い込む||||。
```
