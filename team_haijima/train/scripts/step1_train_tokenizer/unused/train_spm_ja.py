import argparse
import sentencepiece as spm

BASE_PATH = '/persistentshare/storage/team_haijima/dataset_pre'
WIKI_PATH = f'{BASE_PATH}/wiki40b/text_wiki40b.txt'
MC4_PATH = f'{BASE_PATH}/20240405_C4_filtering/text_mC4_skip10.txt'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", action='store_true')
    parser.add_argument("--mc4", action='store_true')
    args = parser.parse_args()
    if args.wiki and args.mc4:
        raise ValueError("Only one of `--wiki` and `--mc4` should be `True`.")
    if not args.wiki and not args.mc4:
        raise ValueError("Either `--wiki` or `--mc4` should be `True`.")
    return args


def train_sentencepiece_tokenizer():
    args = parse_arguments()

    base = 'wiki' if args.wiki else 'mc4'
    model_prefix = f'spm_{base}_ja'

    spm.SentencePieceTrainer.train(
        input=WIKI_PATH if args.wiki else MC4_PATH,
        input_format='text',
        model_prefix=model_prefix,
        model_type='unigram',
        vocab_size=40000,
        character_coverage=0.9995,
        input_sentence_size=200000000,
        num_threads=16,
        max_sentencepiece_length=8,
        split_by_whitespace=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        # user_defined_symbols=[]  # 議論中
        byte_fallback=True,
        normalization_rule_name='nmt_nfkc',
        remove_extra_whitespaces=False,
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        pad_piece='<pad>',
        train_extremely_large_corpus=True,
    )


if __name__ == '__main__':
    train_sentencepiece_tokenizer()
