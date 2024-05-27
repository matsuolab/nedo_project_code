# %%
import glob
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
import random

parser = argparse.ArgumentParser(
    description="Process files")
parser.add_argument('max_workers', type=str,
                    help='Number of parallel workers')
args = parser.parse_args()

# 並列プロセスの数
max_workers = int(args.max_workers)
print("Max workers: ", max_workers)

# 1つのプロセスが処理するファイル数 (modelの読み込みに時間がかかるので､大きめにしておく)
n_file_batch = 10

# %%
with open("temp/gz_list.txt", "r") as f:
    gz_list = f.read().splitlines()

#gz_list = [i for i in gz_list if i.endswith('.gz')]
#gz_list= [i for i in gz_list if 'jawiki' in i]
print(len(gz_list), " files found")

random.shuffle(gz_list)

# %%

dirs = ["temp/fin", "../data/categorized"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)


# %%

def process_files(gz_paths):
    task_list = []
    for gz_path in gz_paths:
        gz_name = gz_path.split("/")[-1]  # .split(".")[0]

        if os.path.exists("temp/fin/" + gz_name):
            print("File already processed")
        else:
            task_list.append(gz_path)

    path_txt = ""
    for gz_path in task_list:
        path_txt += gz_path+" "

    print("Processing ", gz_name)
    # print(path_txt)
    os.system(f"python document_distributor.py {path_txt}")
    with open("temp/fin/" + gz_name, "w") as f:
        f.write("")


def batch_process(gz_list, n_file_batch):
    # リストをn_file_batchのサイズのチャンクに分割
    for i in range(0, len(gz_list), n_file_batch):
        yield gz_list[i:i + n_file_batch]


# ThreadPoolExecutorを使って並列化
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # リストをバッチに分割し、各バッチに対して並列処理を実行
    for gz_batch in batch_process(gz_list, n_file_batch):
        executor.submit(process_files, gz_batch)

# %%
