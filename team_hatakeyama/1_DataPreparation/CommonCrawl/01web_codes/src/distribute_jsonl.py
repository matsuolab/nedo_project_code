
from .classify.Text2Vec import Text2Vec, texts2classes
import os
import json


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def process_lines(docs, t2v, kmeans, base_dir, database_path, check_length=200):
    # docsを処理する関数
    # ここに処理のロジックを実装します
    print(f"Processing {len(docs)} documents...")

    categories = texts2classes(docs, t2v, kmeans, length=check_length)

    for text, category in zip(docs, categories):
        save_dir = f"{base_dir}/{category}"
        make_dir(save_dir)
        database_name = database_path.split("/")[-1]  # .split(".")[0]

        data = json.dumps(
            {
                # "db": database_name, #特に必要ない｡storage 節約
                "text": text},
            ensure_ascii=False
        )
        database_name = database_name.replace(".jsonl", "").replace(".gz", "")
        database_name = database_name.replace(".parquet", "")
        with open(f"{save_dir}/{database_name}.jsonl", "a") as f:
            f.write(data+"\n")

    return len(docs)
