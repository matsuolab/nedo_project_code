from datasets import load_dataset
import json
import os
import gzip

# 設定
save_dir = "../data/original_dump/oscar/"
batch_size = 100000


def load_oscar():
    return load_dataset('oscar', 'unshuffled_deduplicated_ja',
                        split='train',
                        streaming=True
                        )


def save_jsonl_gz(data, filename):
    """指定されたファイル名でgzip圧縮されたJSON Lines形式でデータを保存する"""
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def batch_and_save(dataset):
    """データセットをバッチ処理し、各バッチをgzip圧縮されたファイルに保存する"""
    batch = []
    file_count = 0
    for item in dataset:
        batch.append({"text": item['text']})
        if len(batch) == batch_size:
            save_path = os.path.join(
                save_dir, f'oscar_ja_{file_count}.jsonl.gz')
            save_jsonl_gz(batch, save_path)
            print(f'Saved {save_path}')
            batch = []
            file_count += 1
    # 最後のバッチを保存
    if batch:
        save_path = os.path.join(save_dir, f'oscar_ja_{file_count}.jsonl.gz')
        save_jsonl_gz(batch, save_path)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = load_oscar()
    batch_and_save(dataset)
