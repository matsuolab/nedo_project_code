#!/bin/bash

# 言語を引数から取得します
language=$1

# 出力ファイル名を言語名に設定
output="./datasets/github-code-more-filtering/${language}.jsonl"

# 出力ファイルを初期化します
echo "" > $output

# 各jsonlファイルをループ処理します
for i in $(seq -f "%05g" 0 773)
do
    # jqコマンドを使用して、指定された条件に一致する行を抽出します
    echo train-${i}-of-00774.jsonl
    cat ../github-code-more-filtering/data/jsonl/train-${i}-of-00774.jsonl | jq -c "select(.language==\"${language}\" and (.license==\"mit\" or .license==\"apache-2.0\" or .license==\"bsd-3-clause\"))" >> $output
done

