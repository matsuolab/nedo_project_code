#!/bin/bash

# 出力ファイルを初期化します
echo "" > text_mC4.txt

# 00001.jsonlから01023.jsonlまでの各ファイルをループ処理します
for i in $(seq -f "%05g" 0 1023)
do
  # jqコマンドを使用して"text"フィールドを抽出し、出力ファイルに追加します
  cat ./filtered/c4-ja-${i}-of-01024.json | jq -r '.text' >> text_mC4.txt
done
