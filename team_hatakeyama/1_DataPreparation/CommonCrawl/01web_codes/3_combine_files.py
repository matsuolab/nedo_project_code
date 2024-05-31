import glob
import os
import json
import random
import string
from datetime import datetime, timedelta
from tqdm import tqdm
import time
# 指定したフォルダ内のファイルを検索
n_clusters = 10000
while True:
    for cluster_id in tqdm(range(n_clusters)):
        for i in range(100):
            path_list = glob.glob(f"../data/categorized/{cluster_id}/*.jsonl")
            # 条件に合致するファイルを選ぶ
            eligible_files = []
            for file_path in path_list:
                # ファイルの最終更新時間とサイズをチェック
                stats = os.stat(file_path)
                file_size = stats.st_size
                last_modified = datetime.fromtimestamp(stats.st_mtime)
                if file_size <= 2 * 1024 * 1024 and (datetime.now() - last_modified) > timedelta(minutes=5):
                    eligible_files.append(file_path)

            # 条件に合う2つのファイルをランダムに選び、統合
            if len(eligible_files) >= 2:
                selected_files = random.sample(eligible_files, 2)
                combined_data = []
                for file_path in selected_files:
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        combined_data += lines

                # 新しいランダムなファイル名を生成
                random_name = "00" + \
                    ''.join(random.choices(string.ascii_letters +
                            string.digits, k=10)) + '.jsonl'
                new_path = os.path.join(os.path.dirname(
                    selected_files[0]), random_name)
                combined_data = list(set(combined_data))
                # 統合したデータを新しいファイルに保存
                with open(new_path, "w") as f:
                    f.writelines(combined_data)

                # 元のファイルを削除
                for file_path in selected_files:
                    os.remove(file_path)

    time.sleep(30)
