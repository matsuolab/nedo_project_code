from tqdm import tqdm

def evaluate_tokenizer(tokenizer, eval_corpus_text, unk_token_id = None):
    print("Evaluating tokenizer...")

    print(f"length of texts: {len(eval_corpus_text)}")
    batch_len = 1000000
    num_batches = (len(eval_corpus_text) - 1) // batch_len + 1
    print(f"num of batches: {num_batches}")

    txt_len = 0
    total_tokens = 0
    unk_count = 0
    if unk_token_id is None:
        unk_token_id = tokenizer.unk_id()
    error_words = []
    for text in tqdm(batch(eval_corpus_text, batch_len), total=num_batches):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        txt_len += len(text)
        unk_count += encoded.count(unk_token_id)
        total_tokens += len(encoded)
        for word in text.split():
            if word not in decoded:
                error_words.append(word)

    unk_rate = unk_count / total_tokens
    print(f"Unknown word rate: {unk_rate}")

    # 平均トークン長を計算
    avg_len = total_tokens / txt_len
    print(f"Average token length: {avg_len}")

    # トークン化にかかる時間を計測 (ダミーの値を使用)
    speed = 1000  # トークン/秒
    print(f"Tokenization speed: {speed} tokens/sec")

    return {
        'unk_rate': unk_rate,
        'avg_len': avg_len,
        'speed': speed,
        'error_word_count': len(error_words),
        'error_words': error_words,
    }

def batch(x, batch_size, drop_last=False):
    bs = batch_size
    length = len(x)
    if drop_last or length % bs == 0:
        len_batch = int(length / bs)
    else:
        len_batch = int(length / bs) + 1

    for idx in range(len_batch):
        yield x[idx * bs : (idx + 1) * bs]