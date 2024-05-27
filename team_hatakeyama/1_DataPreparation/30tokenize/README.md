# トークナイザーの学習とtokenize

## [Config](./config.yaml)の編集
- jsonのパスなどを設定する

## トークナイザーの学習
- 6 gbのテキストに対して､ ramは30gbほど使いました｡
- でかすぎると､ramが足りなくなります
- 学習済みモデルは[こちら](./tokenizer/)
~~~
python 1_train_sentencepiece_tokenizer.py
~~~

## トークナイズ
~~~
bash 2_tokenize.sh
~~~


## トークン数の確認
~~~
python count_tokens.py
python count_tokens_parallel.py #並列バージョン

~~~


