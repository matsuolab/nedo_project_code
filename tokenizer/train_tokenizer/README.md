# トークナイザー学習
## 参考：松尾研標準コード
- https://github.com/matsuolab/ucllm_nedo_prod/tree/main/train/scripts/step1_train_tokenizer
- requirement.txtも転用。（トークナイザーの学習のみであれば不要なものも含まれている。）

## ライブラリインストール
```shell
pip install -r requirements.txt
```

## 日本語テキストの場合
- "||||"で事前分割処理したテキストを学習に用いる。
- データは適当な格納先に置いておく。
```shell
python -m train_sentencepiece_tokenizer_unigram_for_ja \
--input ./dataset/wakachi/sample_wakachi.txt \
--model_prefix ja_tokenizer_unigram \
--vocab_size 18000
```
## 英語および数学・プログラムの場合
- データは適当な格納先に置いておく。
```shell
python -m train_sentencepiece_tokenizer_unigram_for_en \
--input ./dataset/sample_en.txt \
--model_prefix en_tokenizer_unigram \
--vocab_size 30000
```
```shell
python -m train_sentencepiece_tokenizer_unigram_for_en \
--input ./dataset/sample_math.txt \
--model_prefix math_tokenizer_unigram \
--vocab_size 500
```
```shell
python -m train_sentencepiece_tokenizer_unigram_for_en \
--input ./dataset/sample_code.txt \
--model_prefix code_tokenizer_unigram \
--vocab_size 500
```
