# 語彙の整備
## 学習したトークナイザーから語彙を取得
- 例）語彙とスコアの一覧をデータフレームにする。
```python
import sentencepiece as spm
import pandas as pd

# 語彙をデータフレーム化
sp = spm.SentencePieceProcessor()
sp.Load(model_path) # SentencePieceのtrainから得られたxxxx.modelを読み込む

vocab_size = sp.get_piece_size()  # 語彙サイズの取得

# トークンとスコアのリストを作成
tokens = []
scores = []

for i in range(vocab_size):
    token = sp.id_to_piece(i)
    score = sp.get_score(i)
    tokens.append(token)
    scores.append(score)

# データフレームの作成
df = pd.DataFrame({
    'token': tokens,
    'score': scores
})

# データフレームの表示
display(df)
```

## 除外・結合・追加・重複削除・語彙数調整などの処理

### 除外
- 日本語の語彙は純粋な日本語のみを取得するため、アルファベットのみで構成される語彙を除外した。

### 結合
- それぞれの言語の語彙を結合。

### 追加
#### 日本語
- 常用漢字や一般語を手動で追加。
    - 参考
        - [常用漢字一覧表](https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kakuki/14/pdf/jyouyou_kanjihyou.pdf)
        - [Wikitinary 日本語](https://ja.wiktionary.org/wiki/%E3%82%AB%E3%83%86%E3%82%B4%E3%83%AA:%E6%97%A5%E6%9C%AC%E8%AA%9E_%E5%90%8D%E8%A9%9E)
        - 日本の地名・東京23区・山手線の駅名など
        - 定型表現・口語
#### 算数
- 数字や記号を追加

### 結合
- データフレームをマージする。

### 重複削除
- 重複したトークンを削除。
```python
df_dedup = df.drop_duplicates(subset="token", keep='first')
```
### 語彙数調整
- 56,320の語彙になるようにスコアの低い語彙を削除する。
- ※事前学習ライブラリの処理の中で、語彙サイズが128の倍数でない場合にダミートークンを追加して128の倍数になるような処理が入っていたことから、初めから128の倍数として設定しておくこととした。（56,320）
- 以下の特殊トークンや連続の空白などを追加することも考慮して最終的な語彙数を調整していく。（参考：[llm-jp-tokneizer](https://github.com/llm-jp/llm-jp-tokenizer/blob/main/scripts/howToCreateModel_ver2.md#%E8%AA%9E%E5%BD%99%E3%81%AE%E3%83%9E%E3%83%BC%E3%82%B8)）
- 最終的な語彙は.vocabという拡張子で保存しておく。
```shell
<unk>
<s>
</s>
<pad>
<CLS>
<SEP>
<EOD>
<MASK>
\n
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁	
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁	
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁▁▁	
▁▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁▁
▁▁▁▁▁▁▁	
▁▁▁▁▁▁
▁▁▁▁▁
▁▁▁▁
▁▁▁
▁▁
▁
```

# トークナイザーの作成
## スコアの再推定
### git clone
- [llm-jp-tokenizer](https://github.com/llm-jp/llm-jp-tokenizer/tree/main)のスクリプトを取得。
```shell
git clone git@github.com:llm-jp/llm-jp-tokenizer.git
```
### 準備
#### スコア再推定用のテキストファイルを格納
- 日本語の事前処理は無しで、各言語を全て統合したテキストファイルをdataディレクトリへ格納する。
- このテキストをもとにUnigramスコアを再推定する。

#### vocabの格納
- 上で最終的な語彙として作成したvocabファイルをmodelディレクトリに格納する。
- このvocabにあるトークンにスコアづけがされる。

### 実行
- SentencePieceをインストール。
```shell
pip install sentencepiece==0.1.99
```
スコア再推定
```shell
python scripts/reestimateScore.py \
    --vocab models/ucllm/sample.vocab \
    --data data/reestimate_all_v6.txt \
    --output models/ucllm/sample.vocab.reestimated \
    --trainingMode EMMT --maxEpoch 1
```
- 特殊トークンとbyte_fall_backのための文字列のスコアを0にする。
- 特殊トークンの最後の位置に注意。ここでは289番目までに設定。
```shell
# 特殊トークンとバイト表記のスコアを0にする
!python scripts/postproc_vocab.py -v models/ucllm/sample.vocab.reestimated --numSpecialTokens 289 > models/ucllm/sample.vocab.reestimated.postproc
```
- llm-jp-tokenizerに従って、ベースとなるダミーのSentencePieceモデルを作っておく。
- 語彙サイズやデータの規模は何でも良い。
```python
import sentencepiece as spm

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
EOD_TOKEN = "<EOD>"
MASK_TOKEN = "<MASK>"
NEWLINE_TOKEN = "\n"

spm.SentencePieceTrainer.train( # SentencePieceのトレーナーを作成
        input="data/sample_reestimate.txt", # インプットファイル
        model_prefix="dummy", # モデルのプレフィックス
        vocab_size=56320, # 語彙サイズ
        character_coverage=0.9995, # 文字カバレッジ
        model_type="unigram", # モデルのタイプ
        num_threads=128, # スレッド数
        add_dummy_prefix=True, # ダミープレフィックスの追加
        train_extremely_large_corpus=True, # 大規模コーパスのトレーニング
        normalization_rule_name='identity', # 正規化の設定 identityは正規化を行わない nfkcは正規化を行う
        user_defined_symbols=[ # ユーザー定義のシンボル
            BOS_TOKEN, # テキストの先頭を表すトークン
            EOS_TOKEN, # テキストの末尾を表すトークン
            PAD_TOKEN, # パディングを表すトークン
            CLS_TOKEN, # 分類タスクの先頭を表すトークン
            SEP_TOKEN, # 分類タスクの区切りを表すトークン
            EOD_TOKEN, # テキストの終わりを表すトークン
            MASK_TOKEN, # マスクを表すトークン
            NEWLINE_TOKEN, # 改行を表すトークン
        ],  # Note: `NEWLINE_TOKEN` is needed in `user_defined_symbols`.
        byte_fallback=True, # 未知語をバイト表現するための設定
        split_digits=True, # 数字を分割するための設定（ここがdefaultだとfalseなので迷う）
        split_by_whitespace=False, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True, # 空白をトークンとして扱うかどうかの設定
        remove_extra_whitespaces=False, # 連続する空白を削除するかどうかの設定
    )
```

## スコア再推定後の語彙からトークナイザーを作成
- スコアを再推定した語彙とダミーのベースモデルを使って、最終的なトークナイザーモデルを作成。
```shell
python scripts/vocab2model.py \ 
    --vocab models/ucllm/sample.vocab.reestimated.postproc \
    --output models/ucllm/sample_new.model \
    --baseModel models/ucllm/dummy.model
```





