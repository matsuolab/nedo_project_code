import json
import pandas as pd

jsonl_file_path = "../data/download/ja2010/train.jsonl"
parquet_file_path = "../data/filter/ja2010/train_0.parquet"
deleted_jsonl_file_path = "../data/deleted/ja2010/deleted_data.jsonl"


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_parquet(file_path):
    return pd.read_parquet(file_path)


def save_jsonl(data, file_path):
    with open(
        file_path, "w", encoding="utf-8"
    ) as file:  # ファイルをUTF-8エンコーディングで開く
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    jsonl_data = load_jsonl(jsonl_file_path)

    parquet_data = load_parquet(parquet_file_path)
    parquet_data = parquet_data.to_dict(orient="records")

    index = {}
    for item in parquet_data:
        index[item["meta"]["curid"]] = 1

    deleted_list = []
    for item in jsonl_data:
        if item["curid"] not in index:
            deleted_list.append(item)

    print(len(jsonl_data), "件中，削除されたデータの数:", len(deleted_list))

    save_jsonl(deleted_list, deleted_jsonl_file_path)


main()
