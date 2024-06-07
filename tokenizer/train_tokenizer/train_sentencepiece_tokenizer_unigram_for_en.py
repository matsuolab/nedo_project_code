import os
import sys
print(f"{os.getcwd() = }")

import argparse
import sentencepiece as spm
# from special_token_list import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, EOD_TOKEN, MASK_TOKEN, NEWLINE_TOKEN

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
EOD_TOKEN = "<EOD>"
MASK_TOKEN = "<MASK>"
NEWLINE_TOKEN = "\n"


# トークナイザーの設定
def parse_arguments():
    parser = argparse.ArgumentParser() # パーサを作る
    parser.add_argument("--input", type=str, required=True) # インプットファイルの指定
    parser.add_argument("--model_prefix", type=str, required=True) # モデルのプレフィックス
    parser.add_argument("--vocab_size", type=int, required=True) # 語彙サイズ
    parser.add_argument("--character_coverage", type=float, default=0.9995) # 文字カバレッジ
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "word", "char"]) # モデルのタイプ
    parser.add_argument("--num_threads", type=int, default=256) # スレッド数
    parser.add_argument("--train_extremely_large_corpus", type=bool, default=True) # 大規模コーパスのトレーニング
    args = parser.parse_args() # 引数を解析
    print(f"{args = }") # 引数を表示
    return args # 引数を返す

# メイン関数
def main():
    args = parse_arguments() # 引数を取得

    # Trains a SentencePiece tokenizer. After training, *.model and *.vocab will be saved in the current directory.
    spm.SentencePieceTrainer.train( # SentencePieceのトレーナーを作成
        input=args.input, # インプットファイル
        model_prefix=args.model_prefix, # モデルのプレフィックス
        vocab_size=args.vocab_size, # 語彙サイズ
        character_coverage=args.character_coverage, # 文字カバレッジ
        model_type=args.model_type, # モデルのタイプ
        num_threads=args.num_threads, # スレッド数
        add_dummy_prefix=True, # ダミープレフィックスの追加
        train_extremely_large_corpus=args.train_extremely_large_corpus, # 大規模コーパスのトレーニング
        normalization_rule_name='identity', # 正規化の設定 identityは正規化を行わない nfkcは正規化を行う
        max_sentencepiece_length=16, # 最大の長さ
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
        split_digits=True, # 数字を分割するための設定
        split_by_whitespace=True, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True, # 空白をトークンとして扱うかどうかの設定
        remove_extra_whitespaces=True # 連続する空白を削除するかどうかの設定,
    )


if __name__ == "__main__":
    main()
