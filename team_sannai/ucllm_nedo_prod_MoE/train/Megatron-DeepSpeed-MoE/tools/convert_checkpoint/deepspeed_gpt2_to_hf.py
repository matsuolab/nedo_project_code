#!/usr/bin/env python

import argparse
import os
import torch
import json
import re

from deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron import _create_rank_checkpoint, parse_arguments

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
# from override_convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from transformers import GPT2Config, LlamaConfig


## for 0.1B
def set_tiny_gpt_config(config: GPT2Config):
    config.vocab_size = 32000
    config.n_positions=1024
    config.n_ctx=1024
    config.n_embd=768
    config.n_layer=12
    config.n_head=12
    config.n_inner=768
    config.bos_token_id=1
    config.eos_token_id=2
    return config

def main():

    # this first part comes mainly from deepspeed_to_megatron.main
    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}')

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp, args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0, args.for_release)

    # the 2nd part comes from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint.main
    # Spell out all parameters in case the defaults change.    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_inner=4096,
        activation_function="gelu",  # used to be "gelu_new" in earlier versions
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )    

    print("model weight")
    lm = input_state_dict["model"]["language_model"]
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]
    for layer_name, _ in transformer.items():
        print(layer_name)

    # Convert.
    print("Converting to HF Checkpoint")
    config = set_tiny_gpt_config(config)
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    
    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)

    # Print the structure of converted state dict.
    #if args.print_checkpoint_structure:
    #    recursive_print(None, output_state_dict)

    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    output_config = config.to_dict()
    output_config["architectures"] = ["GPT2LMHeadModel"]
    output_config["model_type"] = "gpt2"
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)

    print("Now add tokenizer files and upload to the hub")


if __name__ == "__main__":
    main()
