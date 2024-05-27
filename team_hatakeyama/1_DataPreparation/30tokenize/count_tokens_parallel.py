import sentencepiece as spm
import json
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# configファイルの読み込み
with open('config.yaml', 'r') as file:
    conf = yaml.safe_load(file)

# SentencePieceモデルの読み込み
model_path = conf["input_tokenizer_file"]
sp = spm.SentencePieceProcessor(model_file=model_path)

def process_batch(lines):
    # バッチ内の各行を処理
    results = []
    for line in lines:
        try:
            text = json.loads(line)["text"]
            n_tokens = len(sp.encode(text, out_type=str))
            results.append((n_tokens, len(text), 1))  # トークン数、文字数、ドキュメント数
        except json.JSONDecodeError:
            results.append((0, 0, 0))  # JSONのフォーマットエラーがあれば無視
    return results

def init_worker(m_path):
    # ワーカープロセスの初期化時にモデルを読み込む
    global sp
    sp = spm.SentencePieceProcessor(model_file=m_path)

if __name__ == '__main__':
    batch_size = 10000  # 適切なバッチサイズに設定
    with open(conf["input"], "r", encoding="utf-8") as f:
        with Pool(cpu_count(), initializer=init_worker, initargs=(model_path,)) as pool:
            total_tokens, total_length, count = 0, 0, 0
            batch = []
            results = []

            for line in tqdm(f, desc="Reading lines"):
                batch.append(line)
                if len(batch) >= batch_size:
                    results.append(pool.apply_async(process_batch, (batch,)))
                    batch = []

            # 最後のバッチを処理
            if batch:
                results.append(pool.apply_async(process_batch, (batch,)))

            # 結果の集計
            for result in tqdm(results, desc="Processing batches"):
                for tokens, length, cnt in result.get():
                    total_tokens += tokens
                    total_length += length
                    count += cnt

    print("tokens in billion:", total_tokens / 10**9)
    print("total tokens:", total_tokens)
    print("total length:", total_length)
    print("documents:", count)
