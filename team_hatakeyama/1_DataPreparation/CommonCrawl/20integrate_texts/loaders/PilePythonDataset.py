import os
import requests
from datasets import load_dataset


def download_files(base_dir="../data/original_dump/python/"):
    file_range = range(0, 42)  # 0000から0041まで

    # ベースURL
    base_url = "https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/fix-alg-stack/algebraic-stack/train/"

    # ディレクトリが存在しない場合は作成
    os.makedirs(base_dir, exist_ok=True)

    # 指定された範囲のファイルをダウンロード
    for i in file_range:
        filename = f"python{str(i).zfill(4)}.jsonl.zst"
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            response = requests.get(f"{base_url}{filename}")
            # ダウンロードが成功した場合、ファイルを保存
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"{filename} downloaded and saved to {file_path}")
            else:
                print(
                    f"Failed to download {filename}. Status code: {response.status_code}")
        else:
            print(f"{filename} already exists.")

    print("All downloads completed.")


class PilePythonDataset:
    def __init__(self, base_dir, file_pattern="python{:04d}.jsonl.zst"):
        self.base_dir = base_dir
        self.file_pattern = file_pattern
        self.current_file_index = 0
        self.current_loader = None
        download_files(base_dir)
        self.load_next_file()

    def load_next_file(self):
        """次のファイルをロードするメソッド"""
        file_path = os.path.join(
            self.base_dir, self.file_pattern.format(self.current_file_index))
        print("try loading..", file_path)
        if os.path.exists(file_path):
            self.current_loader = iter(load_dataset(
                "json", data_files=file_path, split="train"))
            self.current_file_index += 1
        else:
            self.current_loader = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_loader is None:
            raise StopIteration
        try:
            return next(self.current_loader)
        except StopIteration:
            self.load_next_file()
            if self.current_loader is None:
                raise StopIteration
            return next(self.current_loader)
