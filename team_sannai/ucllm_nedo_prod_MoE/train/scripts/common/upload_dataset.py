"""
データごとにmetadataの形式が異なりmergeされたjsonlファイルをload_dataで読み込めなかったので、
metadataの形式が同じもの毎にload_dataし、concatするようにした。

--jsonl_file_dir: jsonlファイルの存在するディレクトリ
--repo_id: uploadするrepositoryのid
"""

import os
import argparse
from datasets import load_dataset, Features, Value, ClassLabel, Sequence, concatenate_datasets

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args

def main():
    args = parse_arguments()
    jsonl_files = [os.path.join(args.jsonl_file_dir, f) for f in os.listdir(args.jsonl_file_dir) if f.endswith('.jsonl')]
    datasets = []
    for json_file in jsonl_files:
        dataset = load_dataset('json', data_files=json_file, split='train')
        if "version" in dataset.column_names:
            dataset = dataset.remove_columns(["version"])
        if "metadata" in dataset.column_names:
            dataset = dataset.remove_columns(["metadata"])
        datasets.append(dataset)
    ds = concatenate_datasets(datasets)
    ds.push_to_hub(args.repo_id)

if __name__ == "__main__":
    main()
