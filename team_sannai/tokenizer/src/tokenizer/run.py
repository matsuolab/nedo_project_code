import sentencepiece as spm
import os

def train(path: str, name: str, vocab_size: int = 20000, coverage: float = 0.995, model_type: str = 'bpe', max_sentence_length: int = 4096, byte_fallback=False):
    output_path=f"./{name}"
    if not os.path.exists(output_path):
        print("created...", output_path)
        os.makedirs(output_path)
    model_prefix = f"./{output_path}/custom_tokenizer-{name}"

    spm.SentencePieceTrainer.Train(
        input=path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=coverage,
        model_type=model_type,
        max_sentence_length=max_sentence_length,
        byte_fallback=byte_fallback,
        minloglevel=1,
        input_sentence_size=35000000,
        shuffle_input_sentence=True,
    )
    spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

ds_path="/persistentshare/storage/team_sannai/tokenizer/merge_5category_tokenizer_4.0.jsonl"

# ## pattern1
# train(
#     ds_path,
#     "unigram_50000",
#     vocab_size=50000,
#     model_type="unigram"
#     )

## pattern2
train(
    ds_path,
    "unigram_32000",
    vocab_size=32000,
    model_type="unigram"
    )

# ## pattern3
# train(
#     ds_path,
#     "bpe_50000",
#     vocab_size=50000,
#     model_type="bpe"
#     )

## pattern4
train(
    ds_path,
    "bpe_32000",
    vocab_size=32000,
    model_type="bpe"
    )

# ## pattern5
# train(
#     ds_path,
#     "unigram_bf_50000",
#     vocab_size=50000,
#     model_type="unigram",
#     byte_fallback=True
#     )

## pattern6
train(
    ds_path,
    "unigram_bf_32000",
    vocab_size=32000,
    model_type="unigram",
    byte_fallback=True
    )