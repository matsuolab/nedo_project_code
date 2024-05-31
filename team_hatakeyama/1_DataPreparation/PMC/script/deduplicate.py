import os
import asyncio
import json
from pathlib import Path
import Levenshtein
from tqdm import tqdm
import psutil
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

access_token = os.getenv("HF_ACCESS_TOKEN")
repo_id = os.getenv("HF_REPO_ID")

MAX_COMPARE_LENGTH = 128 # 最初のn文字を比較する
SIMILARITY_THRESHOLD = 0.5 # 類似性の閾値

def similarity(s1, s2):
    if not s1 or not s2:
        return 1.0 if s1 == s2 else 0.0
    distance = Levenshtein.distance(s1[:MAX_COMPARE_LENGTH], s2[:MAX_COMPARE_LENGTH])
    max_length = min(max(len(s1), len(s2)), MAX_COMPARE_LENGTH)
    return 1.0 - distance / max_length

async def process_file(file_id, api):
    input_dir = Path("jsonl_files/cluster")
    input_file = input_dir / f"{file_id}.jsonl"
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"dedup_{file_id:04d}.jsonl"
    log_file_path = output_dir / f"log_{file_id:04d}.txt"

    texts_seen = []
    original_lines = 0
    removed_lines = 0

    with open(input_file, 'r') as file, open(output_file, 'w') as out_file, open(log_file_path, 'w') as log_file:
        lines = file.readlines()
        total_lines = len(lines)
        progress = tqdm(total=total_lines, desc=f"Processing {input_file.name}")

        for line in lines:
            doc = json.loads(line)
            original_lines += 1
            if "text" in doc:
                text = doc["text"]
                if not any(similarity(text, seen_text) >= SIMILARITY_THRESHOLD for seen_text in texts_seen):
                    texts_seen.append(text)
                    json.dump(doc, out_file)
                    out_file.write('\n')
                else:
                    removed_lines += 1
            progress.update(1)

        progress.close()

        # ログファイルに統計情報を記録
        log_file.write(f"Original lines: {original_lines}\n")
        log_file.write(f"Removed lines: {removed_lines}\n")

        # メモリ使用量のログ
        memory_usage = psutil.virtual_memory().percent
        log_file.write(f"Final Memory Usage: {memory_usage}%\n")

        # アップロード処理
        api.upload_file(
            token=access_token,
            repo_id=repo_id,
            path_in_repo=f"deduped/{output_file.name}",
            path_or_fileobj=output_file,
            repo_type='dataset'
        )
        print(f"Uploaded {output_file.name} successfully to {repo_id}")

async def main():
    api = HfApi()
    for file_id in range(1024):
        await process_file(file_id, api)
        print(f"File {file_id} processing and upload completed.")

if __name__ == "__main__":
    asyncio.run(main())
