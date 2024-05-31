import os
import json
import logging
import psutil
import sys
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
from scipy.sparse import vstack

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# メモリをチェックして閾値を超えたら終了するようにする。そうしないとインスタンスごと落ちる。
def check_memory_usage(threshold=95):
    memory = psutil.virtual_memory()
    if memory.percent > threshold:
        logging.warning(f"Memory usage exceeded {threshold}%. Exiting...")
        sys.exit(1)

def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

input_file = "jsonl_files/PMC000-010-combined.jsonl"
output_dir = "jsonl_files/cluster"
os.makedirs(output_dir, exist_ok=True)

total_lines = count_lines(input_file)
logging.info(f"Total number of lines in the file: {total_lines}")

vectorizer = HashingVectorizer(n_features=1024, alternate_sign=False, stop_words='english')

n_clusters = 1024
kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++')

def process_file(file_path, batch_size):
    # ファイルをバッチ処理し、テキストデータを読み込んでクラスタリングする
    batch_data = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing lines"):
            json_line = json.loads(line.strip())
            text = json_line['text']
            transformed_text = vectorizer.transform([text])
            batch_data.append((transformed_text, text))
            if len(batch_data) >= batch_size:
                X = vstack([t[0] for t in batch_data])
                kmeans.partial_fit(X)
                predictions = kmeans.predict(X)
                for cluster_id, (_, original_text) in zip(predictions, batch_data):
                    yield cluster_id, original_text
                batch_data = []
                check_memory_usage()

    if batch_data:
        logging.info(f"Processing final batch: {len(batch_data)} texts")
        X = vstack([t[0] for t in batch_data])
        kmeans.partial_fit(X)
        predictions = kmeans.predict(X)
        for cluster_id, (_, original_text) in zip(predictions, batch_data):
            yield cluster_id, original_text
        logging.info("Final batch processed successfully")

batch_size = 1024 * 8
logging.info("Starting clustering and saving...")
for cluster_id, original_text in process_file(input_file, batch_size):
    cluster_file_path = os.path.join(output_dir, f"{cluster_id}.jsonl")
    with open(cluster_file_path, 'a') as file:
        json.dump({"text": original_text}, file)
        file.write('\n')

logging.info("Clustering and saving completed.")
print("Clustering completed and data saved.")
