import os
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import shutil  # 追加

# 環境変数を読み込む
load_dotenv()

# アクセストークンとリポジトリID
access_token = os.getenv("HF_ACCESS_TOKEN")
repo_id = os.getenv("HF_REPO_ID")
repo_type = os.getenv("HF_REPO_TYPE")

# ダウンロードするファイルのリスト
jsonl_files = [
    "PMC000xxxxxx_0.jsonl",
    "PMC001xxxxxx_0.jsonl",
    "PMC002xxxxxx_0.jsonl",
    "PMC003xxxxxx_0.jsonl",
    "PMC004xxxxxx_0.jsonl",
    "PMC005xxxxxx_0.jsonl",
    "PMC006xxxxxx_0.jsonl",
    "PMC007xxxxxx_0.jsonl",
    "PMC008xxxxxx_0.jsonl",
    "PMC009xxxxxx_0.jsonl",
    "PMC010xxxxxx_0.jsonl"
]

# 出力ファイルの設定
output_dir = "jsonl_files"
output_file = "PMC000-010-combined.jsonl"

# tmpディレクトリの作成（存在しない場合）
os.makedirs('tmp', exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ファイルをダウンロードして結合する
with open(os.path.join(output_dir, output_file), 'w') as outfile:
    for filename in jsonl_files:
        print(f"Downloading {filename} to tmp folder...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            cache_dir='tmp',  # ファイルのダウンロード先ディレクトリ
            use_auth_token=access_token
        )
        print(f"{filename} downloaded successfully.")
        
        with open(local_path, 'r') as infile:
            for line in infile:
                outfile.write(line)

        # ローカルのダウンロードファイルを削除
        os.remove(local_path)

# tmpディレクトリのクリーンアップ（変更）
shutil.rmtree('tmp')

print("All datasets have been combined successfully and saved to jsonl_files directory.")