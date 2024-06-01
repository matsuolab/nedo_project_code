手順書:

1. 必要なパッケージをインポートします。
2. 評価用のデータを読み込みます(eval_dataという変数に格納されているものとします)。
3. トークナイザーをロードします(custom_tokenizerとllama_tokenizerという変数に格納されているものとします)。
4. evaluate_tokenizer関数を定義します。この関数はテキストとトークナイザーを受け取り、トークン列、未知語の割合、平均トークン長、処理時間を返します。
5. 空のリストresultsを初期化します。
6. eval_dataの各テキストに対して、evaluate_tokenizer関数を呼び出し、結果をresultsリストに追加します。
7. 結果を表示します。

仕様書:

- evaluate_tokenizer関数
  - 引数:
    - text (str): 評価対象のテキスト
    - tokenizer (Union[LlamaTokenizerFast, sentencepiece.SentencePieceProcessor]): 使用するトークナイザー
  - 返り値:
    - tokens (List[int]): トークン列
    - unk_ratio (float): 未知語の割合
    - avg_length (int): 平均トークン長
    - elapsed_time (float): 処理時間(秒)
- 機能:
  - テキストをトークン化し、未知語の割合、平均トークン長、処理時間を計算する。
  - LlamaTokenizerFastとSentencePieceProcessorの両方に対応する。
- results
  - リスト形式
  - 各要素は以下のキーを持つ辞書:
    - "input" (str): 入力テキスト
    - "custom_perplexity" (float): カスタムトークナイザーのパープレキシティ
    - "custom_time" (float): カスタムトークナイザーの処理時間(秒)
    - "custom_tokens" (List[int]): カスタムトークナイザーのトークン列
    - "custom_unk_ratio" (float): カスタムトークナイザーの未知語の割合
    - "custom_avg_length" (int): カスタムトークナイザーの平均トークン長
    - "llama_perplexity" (float): LlamaTokenizerFastのパープレキシティ
    - "llama_time" (float): LlamaTokenizerFastの処理時間(秒)
    - "llama_tokens" (List[int]): LlamaTokenizerFastのトークン列
    - "llama_unk_ratio" (float): LlamaTokenizerFastの未知語の割合
    - "llama_avg_length" (int): LlamaTokenizerFastの平均トークン長