import json
import pandas as pd

jsonl_file_path = "../data/download/cuX/train.jsonl"
parquet_file_path = "../data/filter/cuX/train_0.parquet"
deleted_jsonl_file_path = "../data/masked/cuX/masked_data.jsonl"


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_parquet(file_path):
    return pd.read_parquet(file_path)


def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    jsonl_data = load_jsonl(jsonl_file_path)

    parquet_data = load_parquet(parquet_file_path)
    parquet_data = parquet_data.to_dict(orient="records")

    text_dict = {}
    for item in jsonl_data:
        text_dict[item["curid"]] = item["text"]

    masked_list = []
    for item in parquet_data:
        if item["text"] != text_dict[item["meta"]["curid"]]:
            masked_list.append(item)

    print(len(jsonl_data), "件中，マスクされたデータの数:", len(masked_list))

    save_jsonl(masked_list, deleted_jsonl_file_path)


main()
