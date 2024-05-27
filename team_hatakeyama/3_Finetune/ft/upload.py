import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging,pipeline
# ログの設定
logging.set_verbosity_info()
logging.enable_default_handler()
logging.enable_explicit_format()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tokenizer_and_model_dir",
                        type=str, required=True)
    parser.add_argument("--huggingface_name", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        input_tokenizer_and_model_dir, device_map="cpu",
        #torch_dtype=torch.float16
        torch_dtype=torch.bfloat16,
        )
    return tokenizer, model


def main() -> None:
    org_name="hatakeyama-llm-team/"
    args = parse_arguments()

    print("loading model")
    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(
        args.output_tokenizer_and_model_dir)

    #pipe=pipeline('text-generation',model=model,tokenizer=tokenizer, 
    #            max_new_tokens=25, 
    #            )

    #inp='以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n将来的な映画製作者が学ぶべき五つの受賞歴のあるドキュメンタリー映画とそれぞれの背景説明を提案してください。\n\n### 応答:\n'
    #gen=
    print("uploading tokenizer")
    local_tokenizer.push_to_hub(org_name+args.huggingface_name)
    print("uploading model")
    local_model.push_to_hub(org_name+args.huggingface_name)
    return


if __name__ == "__main__":
    main()
