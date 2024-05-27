# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import shutil
import json
import os
import re
import sys
import types

import torch

from transformers import LlamaConfig, MistralConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name to the converted checkpoint (Llama2 or Mistral).",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")

    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="bf16",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
}

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.down_proj.weight',
    'mlp.up_proj.weight',
    'mlp.gate_proj.weight'
]


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        print('loading', i, ':', num_checkpoints + 1)
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(path, f"pytorch_model-{i}-of-{num_checkpoints}.bin")
            assert os.path.exists(checkpoint_path), f"Cannot find checkpoint {checkpoint_path}"
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(load_path, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        possible_sub_dir_names = [
            f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}",
            f"mp_rank_{i:02d}_dp_000" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}_dp_000"
        ]
        sub_dir_name = None
        for p in possible_sub_dir_names:
            if os.path.exists(os.path.join(load_path, p)):
                sub_dir_name = p
                break
        assert sub_dir_name is not None, f"Cannot find sub dir in {possible_sub_dir_names}"
        checkpoint_path = os.path.join(load_path, sub_dir_name, 'model_optim_rng.pt')
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    #print(f"path : {path}")
    #print(f"d : {d['model']['language_model'].keys()}")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
        #print(f"k : {k}")
        #print(f"d : {d}")
    return d


def copy_tokenizer(args):
    os.makedirs(args.save_path, exist_ok=True)
    tokenizer_dir = args.load_path
    if os.path.exists(os.path.join(args.load_path, 'tokenizer')):
        tokenizer_dir = os.path.join(args.load_path, 'tokenizer')
    file_list = os.listdir(tokenizer_dir)
    for f in file_list:
        if 'token' in f:
            shutil.copyfile(os.path.join(tokenizer_dir, f), os.path.join(args.save_path, f))


def permute_qkv(
    qkv_w: torch.Tensor, dim: int, n_heads: int, n_heads_kv: int, revert: bool = False
) -> torch.Tensor:

    def permute(x: torch.Tensor) -> torch.Tensor:
        if revert:
            return x.view(head_dim // 2, 2, dim).transpose(0, 1).reshape(head_dim, dim)
        return x.view(2, head_dim // 2, dim).transpose(0, 1).reshape(head_dim, dim)

    head_dim: int = dim // n_heads
    n_qs_per_kv: int = n_heads // n_heads_kv
    n_groups: int = qkv_w.size(0) // head_dim // (n_qs_per_kv + 2)
    groups = torch.chunk(qkv_w, n_groups, dim=0)
    new = []
    for group in groups:
        *qs, k, v = torch.split(group, head_dim, dim=0)
        assert len(qs) == n_qs_per_kv, f"{len(qs)}, {n_qs_per_kv}"
        new += list(map(permute, qs)) + [permute(k), v]
    return torch.cat(new, dim=0)


def convert_wqkv(
    qkv_w: torch.Tensor,  # 7B: [4096x3, 4096]  # type: ignore
    layer_idx: int = 0,
    n_heads: int = 32,
    n_heads_kv: int = 8,
    tp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    llama-2
    qkv_w: 7B: [4096x3, 4096]

    Args:
        qkv_w (torch.Tensor):
        layer_idx (int, optional):
        n_heads (int, optional):
        n_heads_kv (int, optional):

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    n_hidden = qkv_w.size(1)
    hidden_dim: int = n_hidden // n_heads * tp_size
    # qkv_w = permute_qkv(qkv_w, n_hidden, n_heads, n_heads_kv, revert=True)
    #print(f"qkv_w.shape is {qkv_w.shape}")

    n_qs_per_kv: int = n_heads // n_heads_kv
    n_groups: int = qkv_w.size(0) // hidden_dim // (n_qs_per_kv + 2)
    #n_groups = qkv_w.size(0) // (hidden_dim * (n_qs_per_kv + 2))
    qkv_w: list[torch.Tensor] = list(torch.split(qkv_w, hidden_dim, dim=0))
    #qkv_w: list[torch.Tensor] = list(torch.chunk(qkv_w, n_groups, dim=0))
    #print(f"n_groups {n_groups}")
    #print(f"n_qs_per_kv {n_qs_per_kv}")
    #for i in range(len(qkv_w)):
    #    print(f"qkv_w {i} :{qkv_w[i].shape}")
    wq, wk, wv = [], [], []
    for group in range(n_groups):
        for qs in range(n_qs_per_kv):
            wq.append(qkv_w[0])
            del qkv_w[0]
        wk.append(qkv_w[0])
        del qkv_w[0]
        wv.append(qkv_w[0])
        del qkv_w[0]
    assert len(qkv_w) == 0

    wq = torch.concat(wq, dim=0)
    wk = torch.concat(wk, dim=0)
    wv = torch.concat(wv, dim=0)
    return wq, wk, wv


def convert_checkpoint_from_megatron_to_transformers(args: argparse.Namespace) -> None:
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality.

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    release = False
    if 'latest_checkpointed_iteration.txt' in sub_dirs:
        with open(os.path.join(args.load_path, 'latest_checkpointed_iteration.txt')) as f:
            latest_ckpt = f.readline().strip()
            print(f"latest checkpoint: {latest_ckpt}")
            if isinstance(latest_ckpt, bytearray):
                latest_ckpt = latest_ckpt.decode("utf-8")
            try:
                iteration = int(latest_ckpt)
            except ValueError:
                release = (latest_ckpt == "release")
                if not release:
                    raise ValueError(f"Invalid latest checkpoint: {latest_ckpt}")

    else:
        raise ValueError('Cannot find latest ckpt!')
    possible_state_paths: list[str] = [
        os.path.join(
            args.load_path, f"iter_{iteration:07d}" if not release else 'release'  # type: ignore
        )]
    print(f"DEBUG: possible_state_paths: {possible_state_paths}")
    state_path = None
    for p in possible_state_paths:
        if os.path.exists(p):
            state_path = p
            print(f"Loading Megatron-LM checkpoint arguments from: {state_path}")
            break
    assert state_path is not None, f"Cannot find state path in {possible_state_paths}"
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000", "mp_rank_00_dp_000", "mp_rank_00_000_dp_000"]
    state_dirs = os.listdir(state_path)
    for sub_dir in possible_sub_dirs:
        if sub_dir in state_dirs:
            rank0_checkpoint_path = os.path.join(state_path, sub_dir, 'model_optim_rng.pt')
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")  # type: ignore
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")  # type: ignore
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    vocab_size = megatron_args.padded_vocab_size

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
    print(f"dtype : {dtype}")

    if args.model_name == "Llama2":
        config = LlamaConfig(
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            hidden_act='silu',
            hidden_size=megatron_args.hidden_size,
            num_key_value_heads=megatron_args.num_key_value_heads,
            intermediate_size=megatron_args.ffn_hidden_size,
            initializer_range=0.02,
            max_sequence_length=megatron_args.seq_length,
            max_position_embeddings=megatron_args.seq_length,
            model_type='llama',
            num_attention_heads=megatron_args.num_attention_heads,
            num_hidden_layers=megatron_args.num_layers,
            pad_token_id=0,
            rms_norm_eps=megatron_args.layernorm_epsilon,
            torch_dtype=dtype,
            use_cache=True,
            vocab_size=vocab_size,
            architectures=["LLaMAForCausalLM"],
        )
    if args.model_name == "Mistral":
        config = MistralConfig(
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            hidden_act='silu',
            hidden_size=megatron_args.hidden_size,
            num_key_value_heads=megatron_args.num_key_value_heads,
            intermediate_size=megatron_args.ffn_hidden_size,
            initializer_range=0.02,
            max_sequence_length=megatron_args.seq_length,
            max_position_embeddings=megatron_args.seq_length,
            model_type='Mistral',
            num_attention_heads=megatron_args.num_attention_heads,
            num_hidden_layers=megatron_args.num_layers,
            pad_token_id=0,
            rms_norm_eps=megatron_args.layernorm_epsilon,
            torch_dtype=dtype,
            use_cache=True,
            vocab_size=vocab_size,
            architectures=["MistralForCausalLM"],
        )

    print(f"config :{config}")

    output_state_dict = {}

    tp_size: int = megatron_args.tensor_model_parallel_size
    pp_size: int = megatron_args.pipeline_model_parallel_size
    #assert tp_size == 1 and pp_size == 1

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, 0)
    # print(f"DEBUG: pp=0 : tp_state_dicts: {tp_state_dicts[0]}\n\n{tp_state_dicts[1]}\n")
    
    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype).clone().detach().contiguous()
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(state_path, tp_size, pp_size, pp_rank)
        # The transformer.
        path = "model.language_model.encoder"

        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            # Match the name.
            #print(f"val {val}")
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                continue

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("norm"):
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

            # Split QKV packed weights
            elif op_name == "self_attention.query_key_value" and weight_or_bias == "weight":
                print(f"DEBUG: key:{key}, params: {params.shape}")

                wq, wk, wv = convert_wqkv(
                    qkv_w=params, layer_idx=layer_idx, n_heads=config.num_attention_heads,
                    n_heads_kv=config.num_key_value_heads,
                    tp_size=tp_size
                )

                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = wq.to(dtype).clone().detach().contiguous()
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = wk.to(dtype).clone().detach().contiguous()
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = wv.to(dtype).clone().detach().contiguous()

            elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "weight":
                params_per_tp = params.chunk(dim=0, chunks=megatron_args.tensor_model_parallel_size)
                gate = torch.empty(0)
                up = torch.empty(0)
                for t in params_per_tp:
                    gatep, upp = t.chunk(2)
                    gate = torch.cat([gate, gatep])
                    up = torch.cat([up, upp])
                output_state_dict[layer_name + ".mlp.gate_proj.weight"] = gate.to(dtype).clone().detach().contiguous()
                output_state_dict[layer_name + ".mlp.up_proj.weight"] = up.to(dtype).clone().detach().contiguous()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "weight"] = params
                #print(f'out_name {out_name}')
                #print(f'params shape {params.shape}')

            # Copy the bias.
            elif weight_or_bias == "bias":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name + "bias"] = params
                
            rotary_base = 10000
            inv_freq = 1.0 / (rotary_base ** (torch.arange(0, hidden_size_per_head, 2).float() / hidden_size_per_head))
            output_state_dict[layer_name + '.self_attn.rotary_emb.inv_freq'] = inv_freq.to(dtype)
            
    #if config.num_hidden_layers != (layer_idx + 1):  # type: ignore
    #    raise ValueError(f"Expected {config.n_layer} layers but found {layer_idx + 1}")  # type: ignore

    # The final layernorm.
    print(f"Converting final_layernorm")
    #print(params.keys())
    #if pp_size==0:
    #    final_layer_norm_name=layer_idx+1
    #    path="model.language_model.encoder"
    #elif pp_size>1:
    final_layer_norm_name="final_layernorm.weight"
    final_layer_norm_name="layers.32.weight"
    path="model.language_model.encoder"

    print("checking layers for model.language_model.encoder")    
    for key, val in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model.encoder").items():
        print(key)


    #print(get_element_from_dict_by_path(
    #            tp_state_dicts[0], "model.language_model.encoder.final_layernorm"
    #        ))
    final_norm = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], path
            )[final_layer_norm_name]
            for tp_rank in range(1)
        ],
        dim=0
    )
    output_state_dict["model.norm.weight"] = final_norm.to(dtype)

    #print(params.keys())
    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    
    #if pp_size==0:
    #    path="model.language_model.encoder"
    #    embedding_for_head_name="final_layernorm.lm_head.weight"
    #    lm_heads = torch.cat(
    #        [
    #            get_element_from_dict_by_path(
    #                tp_state_dicts[tp_rank], path
    #            )[embedding_for_head_name]
    #            for tp_rank in range(tp_size)
    #        ],
    #        dim=0
    #     )
    #elif pp_size>1:
    path="model"
    print("checking layers for model")    
    for key, val in get_element_from_dict_by_path(tp_state_dicts[0], "model").items():
        print(key)


    embedding_for_head_name="word_embeddings_for_head"
    #embedding_for_head_name="final_layernorm.lm_head.weight"
    #print(get_element_from_dict_by_path(
    #        tp_state_dicts[0], path
    #    )["word_embeddings_for_head"])

    #0502変更
    path="model.language_model.encoder"
    embedding_for_head_name="final_layernorm.lm_head.weight"
    lm_heads = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], path
            )[embedding_for_head_name]#["lm_head.weight"]
            for tp_rank in range(tp_size)
        ],
        dim=0
    )
    #print(f"shape: {lm_heads.shape}")
    output_state_dict["lm_head.weight"] = lm_heads.to(dtype).clone().detach().contiguous()
    ###params[f"final_layernorm.lm_head.weight"].to(dtype)

    # It should be done!
    #print("Conversion from Megatron-LM to Transformers is done!")
    # print(f"DEBUG: tp_state_dicts: {tp_state_dicts[0]}\n\n{tp_state_dicts[1]}\n")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Save tokenizer based on args
    copy_tokenizer(args=args)

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.fs_utils import create_read_file_system  # type: ignore
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    # load the transformers model state dict and config
    sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
    if len(sub_dirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    else:
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    config = LlamaConfig.from_pretrained(args.load_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "ffn_hidden_size": config.intermediate_size,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "GPT2BPETokenizer",
    }

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
    setattr(margs, "params_dtype", dtype)

    # save dummy optim state dict
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,
        "param_groups": [
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],
            }
        ],
    }
    if args.use_distributed_optimizer:
        for i in range(args.target_pipeline_model_parallel_size):
            for j in range(args.target_tensor_model_parallel_size):
                for k in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size == 1:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}"
                    else:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}_{k:03d}"
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )

    # Convert.
    print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # Embedding layer
    print("converting embedding layer")
    # pos_embedding = state_dict["transformer.wpe.weight"].to(dtype)
    word_embedding = state_dict["model.embed_tokens.weight"].to(dtype)
    orig_vocab_size = config.vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    setattr(margs, "padded_vocab_size", padded_vocab_size)
    # Cut out extra padding we don't need
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :]
    # Expanding embedding to larger size by replicating final entry
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
    # Same size!
    else:
        full_word_embed = word_embedding

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i]

    # Transformer layers
    print("converting transformer layers")
    if config.num_hidden_layers % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile(r"model.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"model.layers.{pp_layer_id}.")
            ]

            qkv_weight_to_combine = {}
            mlp_weight_to_combine = {}
            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name.endswith("layernorm"):
                    # out_name = "input_layernorm" if op_name.endswith("1") else "post_attention_layernorm"
                    out_name = op_name
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                elif 'self_attn.o_proj' in op_name and weight_or_bias == 'weight':
                    layer_name = f"layers.{layer}.self_attention.dense.{weight_or_bias}"

                # handle attention K, V, Q weights
                elif op_name.startswith("self_attn") and weight_or_bias == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    # params = params.transpose(0, 1).contiguous()
                    assert (len(qkv_weight_to_combine) != 3)

                    if 'q_proj' in op_name:
                        qkv_weight_to_combine['q_proj'] = params
                    elif 'k_proj' in op_name:
                        qkv_weight_to_combine['k_proj'] = params
                    elif 'v_proj' in op_name:
                        qkv_weight_to_combine['v_proj'] = params

                    if len(qkv_weight_to_combine) == 3:
                        q_weights = qkv_weight_to_combine['q_proj'].chunk(args.target_tensor_model_parallel_size, dim=0)
                        k_weights = qkv_weight_to_combine['k_proj'].chunk(args.target_tensor_model_parallel_size, dim=0)
                        v_weights = qkv_weight_to_combine['v_proj'].chunk(args.target_tensor_model_parallel_size, dim=0)
                        result_weights = []
                        for idx in range(len(q_weights)):
                            partition_weight = torch.cat([q_weights[idx], k_weights[idx], v_weights[idx]])
                            result_weights.append(partition_weight)

                        params = torch.cat(result_weights)
                        layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"
                    else:
                        continue

                elif op_name.startswith("mlp") and weight_or_bias == "weight":
                    if 'down_proj' in op_name:
                        layer_name = f"layers.{layer}.mlp.dense_4h_to_h.{weight_or_bias}"
                    elif 'gate_proj' in op_name:
                        assert (len(mlp_weight_to_combine) != 2)
                        mlp_weight_to_combine['gate_proj'] = params
                    elif 'up_proj' in op_name:
                        assert (len(mlp_weight_to_combine) != 2)
                        mlp_weight_to_combine['up_proj'] = params

                    if 'down_proj' not in op_name and len(mlp_weight_to_combine) == 2:
                        gate_weights = mlp_weight_to_combine['gate_proj'].chunk(args.target_tensor_model_parallel_size, dim=0)
                        up_weights = mlp_weight_to_combine['up_proj'].chunk(args.target_tensor_model_parallel_size, dim=0)
                        result_weights = []
                        for idx in range(len(gate_weights)):
                            partition_weight = torch.cat([gate_weights[idx], up_weights[idx]])
                            result_weights.append(partition_weight)

                        params = torch.cat(result_weights)
                        layer_name = f"layers.{layer}.mlp.dense_h_to_4h.{weight_or_bias}"
                    elif 'down_proj' not in op_name and len(mlp_weight_to_combine) < 2:
                        continue

                else:
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in [
                        "self_attn.o_proj", "mlp.down_proj"] else 0
                    params = torch.chunk(
                        params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(
                        output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone().detach().contiguous() if (op_name + "." + weight_or_bias in tensor_parallel_params)
                        else params.clone().detach().contiguous()
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            params = state_dict[f"model.norm.weight"].to(dtype)
            layer_name = f"final_layernorm.{weight_or_bias}"
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(
                    output_state_dict[i], "model.language_model.encoder")
                params_dict[layer_name] = params.clone().detach().contiguous()

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(
                    output_state_dict[i], "model.lm_head")
                params_dict["weight"] = state_dict['lm_head.weight'].to(
                    dtype).clone().detach().contiguous()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )
            if args.use_distributed_optimizer:
                checkpoint_name = "model_optim_rng.pt"
            else:
                checkpoint_name = "model_optim_rng.pt"
                output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict["optimizer"]
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)

    copy_tokenizer(args=args)


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()