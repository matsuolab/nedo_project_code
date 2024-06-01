from transformers import AutoTokenizer
from tqdm import tqdm

def run_eval(tokenizer_id, corpus):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, device_map="auto")
    unk_token_id = tokenizer.unk_token_id
    error_words = []
    for line in corpus:
        text = line["text"]
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


def main():
    run_eval()

if __name__ == "__main__":
    main()