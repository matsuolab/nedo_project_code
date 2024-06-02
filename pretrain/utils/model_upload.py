from collections import OrderedDict
import json
import torch
from transformers import MixtralForCausalLM, MixtralConfig
from ..models.custom_mixtral import CustomMixtralForCausalLM, CustomMixtralConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seq_length", type=int, default=2048, help="Maximum sequence length")
parser.add_argument("--output_router_logits", action="store_true", help="Whether to output router logits")
parser.add_argument("--bf16", action="store_true", help="Whether to use torch.bfloat16 dtype")
args = parser.parse_args()

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = CustomMixtralConfig.from_dict(config)
    return config

model = CustomMixtralForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None,
    #device_map="auto", 
    attn_implementation="flash_attention_2",
    #max_position_embeddings=args.seq_length,
    #output_router_logits=args.output_router_logits,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    #use_cache=use_cache,
    state_dict=OrderedDict(),
    config = load_config_from_json(config_file = '~/pretrain/config/mixtral_config.json'),
)
print("model set")

model.register_for_auto_class("AutoModelForCausalLM")
model.push_to_hub("TestMoE")
