import sentencepiece as spm

def train(path: str, category: str, vocab_size: int = 20000, coverage: float = 0.995, model_type: str = 'bpe', max_sentence_length: int = 4096, byte_fallback=False):
    # 以下の処理は変更なし
    model_prefix = f"./artifacts/tokenizers/custom_tokenizer-{category}"

    # SentencePiece モデルを学習
    spm.SentencePieceTrainer.Train(
        input=path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=coverage,
        model_type=model_type,
        max_sentence_length=max_sentence_length,
        byte_fallback=byte_fallback
    )

    # 学習したモデルを読み込んでトークナイザーを作成
    tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    return tokenizer

