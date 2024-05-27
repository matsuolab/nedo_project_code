import gzip
import json
import pandas as pd


def read_gzip_json_file(file_path):
    """GZIP圧縮されたJSONファイルから順にテキストデータを読み込むイテレータを生成します。"""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def load_gzip_or_parquet(database_path):
    if database_path.endswith('.gz'):
        lines = []
        for article in read_gzip_json_file(database_path):
            text = article.get('text', '')  # 'text'キーからテキストデータを取得
            lines.append(text)
    # parquetの場合
    elif database_path.endswith('.parquet'):
        df = pd.read_parquet(database_path)
        lines = df['text'].tolist()
    else:
        raise ValueError("Invalid database path", database_path)
    return lines
