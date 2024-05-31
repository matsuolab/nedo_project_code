# %%
import os
import lzma
from tqdm import tqdm
import pandas as pd

# %%
import gzip
import json


def save_jsonl_gz(data, filename):
    """指定されたファイル名でgzip圧縮されたJSON Lines形式でデータを保存する"""
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# %%
path = "../data/original_dump/ja.txt.xz"
save_dir = "../data/original_dump/cc100"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

batch_size = 2000000
batch = []
file_count = 0

with lzma.open(path) as f:
    # lines = f.readlines()
    # for line in tqdm(lines):
    for line in tqdm(f):
        line = line.decode()

        batch.append({"text": line})
        if len(batch) == batch_size:
            save_path = os.path.join(
                save_dir, f'cc100_ja_{file_count}.jsonl.gz')
            save_jsonl_gz(batch, save_path)
            print(f'Saved {save_path}')
            batch = []
            file_count += 1
    # 最後のバッチを保存
    if batch:
        save_path = os.path.join(save_dir, f'cc100_ja_{file_count}.jsonl.gz')
        save_jsonl_gz(batch, save_path)
        print(f'Saved {save_path}')


# %%
len(line)

# %%
