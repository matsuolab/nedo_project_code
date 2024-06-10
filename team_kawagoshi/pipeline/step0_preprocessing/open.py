import gzip
import json

# gzipファイルを開く
with gzip.open('../preprocessed_dataset/filterd_documents/results.filtering.jsonl.gz', 'rt', encoding='utf-8') as f:
    # 各行を処理する
    for line in f:
        # JSON文字列を辞書に変換
        data = json.loads(line)
        
        # データを処理する
        # ここにデータの処理コードを書く
        print(data)