import json
import random
from tqdm import tqdm
from src.load_gz import load_gzip_or_parquet
from src.cleaner.auto_cleaner import ml_clean_text

# クラスタリングに使うデータベースの数
n_gz = 1000
# 各データセットごと､N件のデータを取得
max_articles = 200
max_length = 500
# max_articles = 10
# gzファイルの一覧を取得
out_dir = "temp/texts.jsonl"
with open(out_dir, "w") as f:
    f.write("")

with open("temp/gz_list.txt", "r") as f:
    gz_list = f.readlines()

gz_list = [i for i in gz_list if i != "\n"]
gz_list = [i.replace("\n", "") for i in gz_list]

lines = []
cleaned_text = []

random.shuffle(gz_list)
train_datasets = gz_list[:n_gz]

for path in tqdm(train_datasets):
    print("loading ", path)
    lines = load_gzip_or_parquet(path)

    cnt = 0
    for text in (lines):
        # text=next(dataset_iter)["text"]
        # text = clean_text(text)
        text = text[:max_length]
        text = ml_clean_text(text)

        if text != "":
            cleaned_text.append(text)
            with open(out_dir, "a") as f:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            cnt += 1
            if cnt >= max_articles:
                break
