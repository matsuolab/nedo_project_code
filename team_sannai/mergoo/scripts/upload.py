import argparse
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mergoo.models.modeling_llama import LlamaForCausalLM

from transformers import AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--upload_repo_id", type=str)
    parser.add_argument("--tokenizer", type=str, default="team-sanai/unigram_32000")

    args = parser.parse_args()
    print("args: ", args)
    return args

def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
    )
    new_config = dict({
        "auto_map": dict({
            "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"
        })
    })
    model.config.update(new_config)
    tokenizer.push_to_hub(args.upload_repo_id)
    model.push_to_hub(args.upload_repo_id)

if __name__ == "__main__":
    main()
