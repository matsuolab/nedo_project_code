import os
import time
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

access_token = os.getenv("HF_ACCESS_TOKEN")
repo_id = os.getenv("HF_REPO_ID")

api = HfApi()

directory = "jsonl_files/cluster" # clustering済みのやつをアップロードする場合
file_list = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
progress_bar = tqdm(file_list, desc="Uploading files")

for filename in progress_bar:
    file_path = os.path.join(directory, filename)
    path_in_repo = f"cluster/{filename}" # clustering済みのやつをアップロードする場合

    api.upload_file(
        token=access_token,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        path_or_fileobj=file_path,
        repo_type='dataset'
    )

    progress_bar.set_description(f"Uploading {filename}")
    progress_bar.set_postfix(file=f"Uploaded {filename}")

    # アップロードするファイルの件数が1時間あたり百数十件のような制限があったので、アップロードするファイル数と処理時間に応じてインターバルを挟む場合がある
    # time.sleep(0.1)

print("All files have been uploaded successfully.")
