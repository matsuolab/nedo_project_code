#!/bin/bash

# JSONファイルのパス
input_json="cc100-ja.jsonl"
# 分割後のファイルが保存されるディレクトリ
output_dir="output_cc100_ja"

# 出力ディレクトリがなければ作成
mkdir -p "$output_dir"

# ファイルを10000行ごとに分割されたファイルに分割し、ファイル名-00.json, ファイル名-01.json, ... の形式で出力
split -l 10000 --additional-suffix=.json "$input_json" "$output_dir/cc100-ja-"
sed -i '/{"text": "\\n"}/d' "$output_dir/cc100-ja-*"

# split コマンドは通常 xaa, xab, ... のようにファイルを分割するが、
# ファイル名が連番になるようにリネームする
# count=0
# for file in "$output_dir/"cc100-ja-*; do
#   mv "$file" "$output_dir/cc100-ja-$(printf "%02d" $a).json"
#   let count=count+1
# done
