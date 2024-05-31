# %%
# クリーニングスクリプトに入れ忘れた、日付系を消す処理
import glob
import json
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import random
# 冒頭の数字だけの行を削除


def remove_num_lines(record):
    lines = record["text"].split("\n")
    new_lines = []
    for line in lines:
        if len(line) == 0:
            continue
        check_line = line[:20]
        count = sum(c.isdigit() for c in check_line)
        # num_ratio=count/len(check_line)
        num_ratio = count
        # print(num_ratio)
        ratio = 5
        if num_ratio > ratio and check_line.find(":") > 0:
            continue
        if num_ratio > ratio and check_line.find("日") > 0:
            continue
        if num_ratio > ratio and check_line.find("年") > 0:
            continue
        if num_ratio > ratio and check_line.find("-") > 0:
            continue
        if num_ratio > ratio and check_line.find("/") > 0:
            continue
        if num_ratio > ratio and check_line.find("／") > 0:
            continue
        if num_ratio > ratio and check_line.find("月") > 0:
            continue

        new_lines.append(line)
    record["text"] = "\n".join(new_lines)
    return record


cluster_ids = list(range(10000))
random.shuffle(cluster_ids)
for cat_id in tqdm(cluster_ids):
    jsonl_list = glob.glob(
        f"../data/categorized/{cat_id}/*.jsonl", recursive=True)
    for path in (jsonl_list):
        # ファイルの最終更新時間とサイズをチェック
        try:
            stats = os.stat(path)
        except:
            continue
        last_modified = datetime.fromtimestamp(stats.st_mtime)
        if (datetime.now() - last_modified) > timedelta(minutes=5):
            # if True:

            record_list = []
            with open(path, "r") as f:
                for record in f:
                    data = json.loads(record)
                    record_list.append(data)

            cleaned_record_list = []
            for record in record_list:
                cleaned = remove_num_lines(record)
                if len(cleaned["text"]) > 0:
                    cleaned_record_list.append(cleaned)

            with open(path, "w") as f:
                for record in cleaned_record_list:
                    f.write(json.dumps(record, ensure_ascii=False)+"\n")
            # break
            # if path=="../data/categorized/39/00A2Zpve0kEr.jsonl":
            #    break

# %%
