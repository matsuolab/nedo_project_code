import argparse
import os
from transformers import AutoTokenizer, T5Tokenizer, LlamaTokenizer

import sys

ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(os.path.join(ROOT_PATH, "common"))

from special_token_list import UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, EOD_TOKEN, MASK_TOKEN


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_type", type=str, default="LlamaTokenizer")
    parser.add_argument("--input_tokenizer_file", type=str, required=False, default="~/nedo_project_code/team_kawagoshi/pipeline/step3_tokenization/output/botchan.model")
    parser.add_argument("--output_model_name", type=str, required=False, default="test_tokenizer")
    parser.add_argument("--output_tokenizer_dir", type=str, required=False, default="~/nedo_project_code/team_kawagoshi/pipeline/step3_tokenization/output/tokenizer")
    parser.add_argument("--test_prompt_text", type=str, default="Once upon a time,")
    args = parser.parse_args()
    print(f"{args = }")
    return args

def main() -> None:
    args = parse_arguments()

    # Converts the tokenizer from SentencePiece format to HuggingFace Transformers format by loading with `T5Tokenizer`.
    # Note: `PreTrainedTokenizerFast` (base class) doesn't support byte fallback, but `T5Tokenizer` (derived class) supports byte fallback
    # https://zenn.dev/selllous/articles/transformers_pretrain_to_ft#tokenizers-t5tokenizer%E5%BD%A2%E5%BC%8F%E3%81%B8%E3%81%AE%E5%A4%89%E6%8F%9B
    if args.tokenizer_type == "T5Tokenizer":
        applied_tokenier = T5Tokenizer
    elif args.tokenizer_type == "LlamaTokenizer":
        applied_tokenier = LlamaTokenizer

    output_tokenizer = applied_tokenier(
        vocab_file=os.path.expanduser(args.input_tokenizer_file),
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
        additional_special_tokens=[
            EOD_TOKEN,
        ],  # Note: `NEWLINE_TOKEN` is NOT needed in `additional_special_tokens`.
        extra_ids=0,
        model_max_length=2048,  # TODO: Remove hard coding and/or magic number.
        split_special_tokens=True,
    )

    output_tokenizer_dir = os.path.expanduser(args.output_tokenizer_dir)
    os.makedirs(output_tokenizer_dir, exist_ok=True)

    output_tokenizer.save_pretrained(output_tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_tokenizer_dir)
    tokenizer.push_to_hub(args.output_model_name)

if __name__ == "__main__":
    main()
