#!/usr/bin/env python

import argparse
import os
import torch
import json
import re

from deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron import _create_rank_checkpoint

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
# from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint, fix_query_key_value_ordering
from transformers import LlamaConfig

## for 0.1B
def set_small_config(config: LlamaConfig):
    config.vocab_size = 32000
    config.hidden_size = 768
    config.intermediate_size = 768
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.num_key_value_heads = 4
    config.max_position_embeddings = 1024
    config.bos_token_id=1
    config.eos_token_id=2
    config.ffn_hidden_size = 768
    return config

## for 350M
def set_mid_config(config: LlamaConfig):
    config.vocab_size = 32000
    config.hidden_size = 1024 # 768
    config.intermediate_size = 1024 #768
    config.num_hidden_layers = 24 # 12
    config.num_attention_heads = 16
    config.num_key_value_heads = 4
    config.max_position_embeddings = 1024
    config.bos_token_id=1
    config.eos_token_id=2
    config.ffn_hidden_size = 1024 #768
    return config

## for 760M
def set_large_config(config: LlamaConfig):
    config.vocab_size = 32000
    config.hidden_size = 1536 # 768
    config.intermediate_size = 1536 #768
    config.num_hidden_layers = 24 # 12
    config.num_attention_heads = 16
    config.num_key_value_heads = 8 # 4
    config.max_position_embeddings = 1024
    config.bos_token_id=1
    config.eos_token_id=2
    config.ffn_hidden_size = 1536 #768
    return config
def _fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        # save_shae (12, 3, 64, 768)
        # params torch.Size([1280, 768])
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

## https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py#L94
def _convert_megatron_checkpoint(args, input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        # from pprint import pprint
        # pprint(vars(ds_args))
        print("ds_args", ds_args)
        config.vocab_size = ds_args.padded_vocab_size
        config.n_positions = ds_args.max_position_embeddings
        config.n_embd = ds_args.hidden_size
        config.n_layer = ds_args.num_layers
        config.n_head = ds_args.num_attention_heads
        config.n_inner = ds_args.ffn_hidden_size
        config.rms_norm_eps = ds_args.layernorm_epsilon
        config.rope_theta = 10000 ## megatron-deepspeedでは10000で固定

        # pprint(config)

    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head

    # The key heads num
    num_key_heads = config.num_key_value_heads
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # The position embeddings.
    # pos_embeddings = embeddings["position_embeddings"]["weight"]

    # Read the causal mask dimension (seqlen). [max_sequence_length, hidden_size]
    # n_positions = pos_embeddings.size(0)
    # if n_positions != config.n_positions:
    #     raise ValueError(
    #         f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
    #     )
    # Store the position embeddings.
    #output_state_dict["transformer.wpe.weight"] = pos_embeddings
    n_positions = config.n_positions

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    print("megatron model weight")
    for key, val in transformer.items():
        print(key, val.shape)

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"model.layers.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if (op_name == "input_layernorm" or op_name == "post_attention_layernorm"):
            output_state_dict[layer_name + "." + op_name + "." + weight_or_bias] = val
        # Transpose the QKV matrix.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # Insert a tensor of 1x1xDxD bias.
            #causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
            #    1, 1, n_positions, n_positions
            #)
            # output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # Insert a "dummy" tensor for masked_bias.
            # masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            # output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            # out_val = _fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            query_hidden_num = hidden_size_per_head*heads
            key_value_hidden_num = hidden_size_per_head*num_key_heads
            (query_weights, key_weights, value_weights) = torch.split(val, [query_hidden_num, key_value_hidden_num, key_value_hidden_num], dim=0)

            # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
            # out_val = out_val.transpose(0, 1).contiguous()
            # Store.
            # output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val

            output_state_dict[layer_name + ".self_attn.q_proj.weight"] = query_weights
            output_state_dict[layer_name + ".self_attn.k_proj.weight"] = key_weights
            output_state_dict[layer_name + ".self_attn.v_proj.weight"] = value_weights
        elif(
            op_name == "self_attention.dense"
        ):            
            output_state_dict[layer_name + ".self_attn.o_proj.weight"] = val
        elif(op_name == "mlp.dense_h_to_4h"):            
            (gate_weights, up_proj_weights) = torch.split(val, [config.n_inner, config.n_inner], dim=0)
            output_state_dict[layer_name + ".mlp.gate_proj.weight"] = gate_weights
            output_state_dict[layer_name + ".mlp.up_proj.weight"] = up_proj_weights
        elif(op_name == "mlp.dense_4h_to_h"):
            output_state_dict[layer_name + ".mlp.down_proj.weight"] = val
        # Transpose the bias.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            out_val = _fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Store. No change of shape.
            output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val
        # Transpose the weights.
        elif weight_or_bias == "weight":
            pass
            # out_name = megatron_to_transformers[op_name]
            # output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":
            pass
            #out_name = megatron_to_transformers[op_name]
            #output_state_dict[layer_name + out_name + "bias"] = val

    # DEBUG.
    assert config.n_layer == layer_idx + 1

    # The final layernorm.
    #output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    #output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    # output_state_dict["lm_head.weight"] = word_embeddings

    output_state_dict["model.norm.weight"] = transformer[f"layers.{layer_idx + 1}.weight"]
    output_state_dict["lm_head.weight"] = transformer["final_layernorm.lm_head.weight"]

    # It should be done!
    return output_state_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input DeepSpeed Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output Megatron checkpoint folder')
    parser.add_argument('--target_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--target_pp', default=1, type=int, help='Target PP degree')
    parser.add_argument('--for_release', action='store_true', help='Convert for release purpose, reset some (progress) counters.')    
    args = parser.parse_args()
    print(f'args = {args}')
    return args

def main():

    # this first part comes mainly from deepspeed_to_megatron.main
    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}')

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp, args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    input_state_dict = _create_rank_checkpoint(ds_checkpoint, args.target_tp - 1, args.target_pp - 1, args.for_release)

    # the 2nd part comes from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint.main
    # Spell out all parameters in case the defaults change.
    config = LlamaConfig()
    # llama2のモデルサイズで手作業で変更
    # config = set_small_config(config)
    config = set_mid_config(config)
    # config = set_large_config(config)
    

    # print("model weight")
    lm = input_state_dict["model"]["language_model"]
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]
    # for layer_name, _ in transformer.items():
    #     print(layer_name)
    # print("="*20)

    # Convert.
    print("Converting to HF Checkpoint")
    output_state_dict = _convert_megatron_checkpoint(args, input_state_dict, config)

    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)

    # Print the structure of converted state dict.
    #if args.print_checkpoint_structure:
    #    recursive_print(None, output_state_dict)

    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    output_config = config.to_dict()
    output_config["architectures"] = ["LlamaForCausalLM"]
    output_config["model_type"] = "llama"
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
