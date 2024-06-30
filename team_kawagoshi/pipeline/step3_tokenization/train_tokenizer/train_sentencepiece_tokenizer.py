# Appends a path to import python scripts that are in other directories.
import os
import sys
import argparse
import sentencepiece as spm


def parse_arguments():
    parser = argparse.ArgumentParser()
    # gzは解凍しておく
    parser.add_argument("--input", type=str, required=False, default="../step00_download_datasets/output/refinedweb/refinedweb.jsonl")
    parser.add_argument("--output_base", type=str, required=False, default="./output")
    parser.add_argument("--model_prefix", type=str, required=False, default="botchan")
    parser.add_argument("--vocab_size", type=int, required=False, default=60000)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "word", "char"])
    parser.add_argument("--num_threads", type=int, default=200)
    parser.add_argument("--train_extremely_large_corpus", type=bool, default=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def main():
    args = parse_arguments()
    if not os.path.exists(args.output_base):
        os.makedirs(args.output_base)
        
    model_prefix = os.path.join(args.output_base, args.model_prefix)
    NEWLINE_TOKEN = "\n"

    # Trains a SentencePiece tokenizer. After training, *.model and *.vocab will be saved in the current directory.
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        num_threads=args.num_threads,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
        normalization_rule_name='identity',
        user_defined_symbols=[
            NEWLINE_TOKEN,
        ],  # Note: `NEWLINE_TOKEN` is needed in `user_defined_symbols`.
        max_sentencepiece_length=16,
        byte_fallback=True,
        split_digits=True,
        split_by_whitespace=False,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False
    )


if __name__ == "__main__":
    main()
