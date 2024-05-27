# %%
from huggingface_hub import HfApi, logging
import os
import glob
import json
import pandas as pd
import time
from tqdm import tqdm
# os.environ["HF_ENDPOINT"] = "http://localhost:5564"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
logging.set_verbosity_debug()
hf = HfApi()

# %%
n_clusters = 10000

done_list_path = "../data/temp/done_list.txt"


def load(cluster_id):

    files = glob.glob(f"../data/dedup_categorized/{cluster_id}/*.jsonl")
    if len(files) == 0:
        return []
    text_list = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                text_list.append(json.loads(line)["text"])

    text_list = list(set(text_list))
    return text_list


def upload(cnt, text_list):
    df = pd.DataFrame(text_list, columns=["text"])
    df.to_parquet(f"../data/temp/temp.parquet")
    try:
        print(f"Uploading {cnt}")
        hf.upload_file(path_or_fileobj="../data/temp/temp.parquet",
                       path_in_repo=f"parquet_files/{cnt}.parquet",
                       repo_id="kanhatakeyama/TanukiCorpus", repo_type="dataset")
        time.sleep(100)
    except Exception as e:
        print(e)


text_list = []
cnt = 0
for cluster_id in tqdm(range(n_clusters)):
    text_list += load(cluster_id)
    if len(text_list) > 3*10**6:
        upload(cnt, text_list)
        text_list = []
        cnt += 1
cnt += 1
upload(cnt, text_list)
