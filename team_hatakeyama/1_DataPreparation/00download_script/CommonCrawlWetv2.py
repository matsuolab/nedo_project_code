# %%

import glob
from datasets import load_dataset
import json
import os
import gzip

# 設定
save_dir = "../data/original_dump/commoncrawl/2021gz"
batch_size = 100000


def load(parquet_id):
    return load_dataset('hatakeyama-llm-team/CommonCrawl_wet_v2',
                        split='train',
                        data_files=f"https://huggingface.co/datasets/hatakeyama-llm-team/CommonCrawl_wet_v2/resolve/main/CC-MAIN-2021-04/{parquet_id}.parquet",
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
    except Exception as e:
        print(f"Failed to load {parquet_id}")
        print(e)
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
                save_dir, f'cc2021_{parquet_id}_{file_count}.jsonl.gz')
            save_jsonl_gz(batch, save_path)
            print(f'Saved {save_path}')
            batch = []
            file_count += 1
    # 最後のバッチを保存
    if batch:
        save_path = os.path.join(
            save_dir, f'cc2021_{parquet_id}_{file_count}.jsonl.gz')
        save_jsonl_gz(batch, save_path)
        print(f'Saved {save_path}')

    return True


# %%
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# batch_and_save(dataset)

# 生成する文字列の数
# num_strings = 9999
# ids= [f"{i:04}" for i in range(1, num_strings + 1)]
parq_list = glob.glob(
    "../data/original_dump/commoncrawl/CommonCrawl_wet_v2/CC-MAIN-2021-04/batch*.parquet")
parq_list = [os.path.basename(p).split(".")[0] for p in parq_list]
parq_list


# %%

for parquet_id in parq_list:
    print(parquet_id)
    if not batch_and_save(parquet_id):
        break


# %%
