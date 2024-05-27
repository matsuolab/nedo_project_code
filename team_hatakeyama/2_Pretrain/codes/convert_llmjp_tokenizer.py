#https://github.com/llm-jp/llm-jp-tokenizer/blob/main/hf/convert_llmjp_unigram_spm_to_hf_fast.py

from tokenizers import decoders, models, normalizers, processors, Regex, Tokenizer
import argparse
import sys
import os
sys.path.append("../common")
from special_token_list import *
TOKENIZER_CONFIG_JSON = """{
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "3": {
      "content": "<pad>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "4": {
      "content": "<CLS>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "5": {
      "content": "<SEP>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "6": {
      "content": "<EOD>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "7": {
      "content": "<MASK>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "additional_special_tokens": [
    "<EOD>"
  ],
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "cls_token": "<CLS>",
  "eos_token": "</s>",
  "extra_ids": 0,
  "mask_token": "<MASK>",
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "<pad>",
  "sep_token": "<SEP>",
  "sp_model_kwargs": {},
  "tokenizer_class": "PreTrainedTokenizerFast",
  "unk_token": "<unk>"
}
"""


SPECIAL_TOKENS_MAP_JSON = """{
  "additional_special_tokens": [
    "<EOD>"
  ],
  "bos_token": {
    "content": "<s>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "cls_token": {
    "content": "<CLS>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "eos_token": {
    "content": "</s>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "mask_token": {
    "content": "<MASK>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "pad_token": {
    "content": "<pad>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "sep_token": {
    "content": "<SEP>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "unk_token": {
    "content": "<unk>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  }
}
"""


def format_special_token(label: str):
    return label
    return f"{label[:-1]}{label[-1]}"


def get_proto():
    try:
        import sys

        sys.path.append(".")

        import sentencepiece_model_pb2 as model
    except Exception:
        raise Exception(
            "You don't seem to have the required protobuf file, in order to use this function you need to run `pip install protobuf` and `wget https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_model_pb2.py` for us to be able to read the intrinsics of your spm_file. `pip install sentencepiece` is not required."
        )

    m = model.ModelProto()
    return m


def convert_llmjp_unigram_spm_to_hf(input_sp_model_path: str, eod_token: str) -> Tokenizer:
    proto = get_proto()
    proto.ParseFromString(open(input_sp_model_path, "rb").read())
    model_type = proto.trainer_spec.model_type
    assert model_type == 1, f"You're trying to run a `Unigram` model but you're file was trained with a different algorithm ({model_type=})"
    vocab = [(piece.piece, piece.score) for piece in proto.pieces if piece.piece != ""]
    unk_id = proto.trainer_spec.unk_id
    special_tokens = [_ for _, piece in enumerate(proto.pieces) if piece.type in [2, 3, 4, 5]]
    for _, token_id in enumerate(special_tokens):
        vocab[token_id] = format_special_token(vocab[token_id][0]), vocab[token_id][1]
        special_tokens[_] = vocab[token_id][0]
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id, byte_fallback=True))
    tokenizer.add_special_tokens(special_tokens)
    normalizer_list = []
    precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
    if precompiled_charsmap:
        normalizer_list.append(normalizers.Precompiled(precompiled_charsmap))
    replacement = "▁"
    """
    # do not use Metaspace pre_tokenizer because all the continuous spaces are divided into single space sequences 
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=replacement, add_prefix_space=True
    )
    """
    # using normalizer to insert "▁" to the beginning of text and to replace space to "▁"
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("(?<!\\n)^"), replacement),
            normalizers.Replace(Regex(" "), replacement),
        ]
    )
    eod = format_special_token(eod_token)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=["$0", eod],
        pair=["$A", eod, "$B:1", f"{eod}:1"],
        special_tokens=[
            (eod, tokenizer.get_vocab()[eod]),
        ],
    )
    """
    # do not use Metaspace decoder because all the heading spaces are removed
    tokenizer.decoder = decoders.Metaspace(
        replacement=replacement, add_prefix_space=True
    )
    """
    # using Replace decoders to remove the extra space char at the beginning of text and replace "▁" to space
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Replace(Regex(replacement), " "),
            decoders.Fuse(),
            decoders.Replace(Regex(f"(?<!\\n)^ "), ""),
        ]
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_sp_model_path",
        required=True,
        type=str,
        help="path for input sentencepiece unigram model file",
    )
    parser.add_argument(
        "-o", "--output_hf_tokenizer_dir",
        required=True,
        type=str,
        help="path for output huggingface tokenizers directory",
    )
    parser.add_argument(
        "-e", "--eod_token",
        #default="<EOD>",
        default="</s>",
        type=str,
        help="the end-of-sentence token which appended to the results of encode(), default='</s>'",
    )
    args = parser.parse_args()
    print("converting", args.input_sp_model_path, "to", args.output_hf_tokenizer_dir)
    os.makedirs(args.output_hf_tokenizer_dir, exist_ok=True)
    tokenizer_json_path = os.path.join(args.output_hf_tokenizer_dir, "tokenizer.json")
    tokenizer_config_json_path = os.path.join(args.output_hf_tokenizer_dir, "tokenizer_config.json")
    special_tokens_map_json_path = os.path.join(args.output_hf_tokenizer_dir, "special_tokens_map.json")
    tokenizer = convert_llmjp_unigram_spm_to_hf(args.input_sp_model_path, args.eod_token)
    #tokenizer = convert_llmjp_unigram_spm_to_hf(args.input_sp_model_path, args.eod_token)
    tokenizer.save(tokenizer_json_path)
    with open(tokenizer_config_json_path, "w", encoding="utf8") as fout:
        print(TOKENIZER_CONFIG_JSON, file=fout)
    with open(special_tokens_map_json_path, "w", encoding="utf8") as fout:
        print(SPECIAL_TOKENS_MAP_JSON, file=fout)


    ###########
    # \nトークを消さないと､改行後に空白が入る
    print("manually removing \\n token (id=8) ")
    import json
    with open(tokenizer_json_path, "r", encoding="utf8") as fin:
        tokenizer_json =json.loads(fin.read())

    #print(tokenizer_json["added_tokens"])
    tokenizer_json["added_tokens"].pop(8)
    json.dump(tokenizer_json, open(tokenizer_json_path, "w", encoding="utf8"), indent=2)

if __name__ == "__main__":
    main()