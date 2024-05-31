# %%
from tqdm import tqdm
import os
import json
import gzip
import pandas as pd
path = "../data/original_dump/dataset.parquet"
save_dir = "../data/original_dump/shisa"

df = pd.read_parquet(path)

# %%


def save_jsonl_gz(data, filename):
    """指定されたファイル名でgzip圧縮されたJSON Lines形式でデータを保存する"""
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# %%
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

batch_size = 100000
batch = []
file_count = 0
lines = list(df["text"])
for line in tqdm(lines):
    # line=line.decode()

    batch.append({"text": line})
    if len(batch) == batch_size:
        save_path = os.path.join(
            save_dir, f'{file_count}.jsonl.gz')
        save_jsonl_gz(batch, save_path)
        print(f'Saved {save_path}')
        batch = []
        file_count += 1
# 最後のバッチを保存
if batch:
    save_path = os.path.join(save_dir, f'{file_count}.jsonl.gz')
    save_jsonl_gz(batch, save_path)
    print(f'Saved {save_path}')


# %%
lines[:3]

# %%
