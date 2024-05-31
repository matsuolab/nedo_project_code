

from datasets import load_dataset
import json
import os
import gzip

# 設定
save_dir = "../data/original_dump/jap2010/"
batch_size = 10000


def load(parquet_id):
    # streamingだと､なぜか15k行目くらいでerrorが出てくる
    return load_dataset('hatakeyama-llm-team/japanese2010',
                        split='train',
                        data_files=f"https://huggingface.co/datasets/hatakeyama-llm-team/japanese2010/resolve/refs%2Fconvert%2Fparquet/default/train/{parquet_id}.parquet"
                        # streaming=True
                        )


def save_jsonl_gz(data, filename):
    """指定されたファイル名でgzip圧縮されたJSON Lines形式でデータを保存する"""
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def batch_and_save(parquet_id):
    try:
        dataset = load(parquet_id)
    except:
        print(f"Failed to load {parquet_id}")
        return False
    """データセットをバッチ処理し、各バッチをgzip圧縮されたファイルに保存する"""
    batch = []
    file_count = 0
    # for item in dataset:
    for i in range(len(dataset)):
        try:
            text = dataset[i]['text']
            batch.append({"text": text})
        except Exception as e:
            print(e)
        continue

        if len(batch) == batch_size:
            save_path = os.path.join(
                save_dir, f'jap2010_{parquet_id}_{file_count}.jsonl.gz')
            save_jsonl_gz(batch, save_path)
            print(f'Saved {save_path}')
            batch = []
            file_count += 1
    # 最後のバッチを保存
    if batch:
        save_path = os.path.join(
            save_dir, f'jap2010_{parquet_id}_{file_count}.jsonl.gz')
        save_jsonl_gz(batch, save_path)
        print(f'Saved {save_path}')

    return True


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# batch_and_save(dataset)

# 生成する文字列の数
num_strings = 9999
ids = [f"{i:04}" for i in range(1, num_strings + 1)]

for parquet_id in ids:
    print(parquet_id)
    if not batch_and_save(parquet_id):
        break
"""
#streamingだと､途中でエラーが出た｡
from datasets import load_dataset
import json
import os
import gzip

# 設定
save_dir = "../data/original_dump/jap2010/"
batch_size = 10000


def load():
    return load_dataset('hatakeyama-llm-team/japanese2010',
                        split='train',
                        streaming=True
                        )


def save_jsonl_gz(data, filename):
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def batch_and_save(dataset):
    batch = []
    file_count = 0
    for item in dataset:
        batch.append({"text": item['text']})
        if len(batch) == batch_size:
            save_path = os.path.join(
                save_dir, f'jap2010_{file_count}.jsonl.gz')
            save_jsonl_gz(batch, save_path)
            print(f'Saved {save_path}')
            batch = []
            file_count += 1
    # 最後のバッチを保存
    if batch:
        save_path = os.path.join(save_dir, f'jap2010_{file_count}.jsonl.gz')
        save_jsonl_gz(batch, save_path)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = load()
    batch_and_save(dataset)
"""
