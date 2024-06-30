from transformers import AutoTokenizer, GPT2TokenizerFast
import argparse
import os

def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(input_tokenizer_and_model_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
  
    return tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_dir", type=str, required=False, default="Xenova/gpt-4")
    parser.add_argument("--output_tokenizer_dir", type=str, required=False, default="~/nedo_project_code/team_kawagoshi/pipelinestep3_tokenization/output/download_tokenizer")
    args = parser.parse_args()
    print(f"{args = }")
    return args

def main() -> None:
    args = parse_arguments()
    # Loads and tests the local tokenizer and the local model.
    tokenizer = load_tokenizer_and_model(args.input_tokenizer_dir)
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)
    vocab_len = len(vocab)
    print(vocab_len)
    tokenizer.save_pretrained(os.path.expanduser(args.output_tokenizer_dir))

if __name__ == "__main__":
    main()