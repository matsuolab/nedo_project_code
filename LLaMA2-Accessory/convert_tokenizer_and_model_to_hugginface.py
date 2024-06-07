import json
import os
from typing import Any, Dict, List
import argparse

import torch

try:
    import transformers
except ImportError:
    raise NotImplementedError("transformers must be installed before "
                              "converting the weights.")
print("transformers version:", transformers.__version__)
hf_major_ver, hf_minor_ver = [
    int(value) for value in transformers.__version__.split(".")[:2]
]
if (hf_major_ver, hf_minor_ver) < (4, 31):
    raise NotImplementedError("Requires transformers >= 4.31.0 to convert the "
                              "weights.")


def convert_merged_ckpt_to_hf(
    merged_state_dict: Dict[str, torch.Tensor],
    params: Dict[str, Any],
    model_type: str = 'llama'
) -> List[Dict[str, torch.Tensor]]:

    merged_state_dict = merged_state_dict.copy()
    num_layers = 0
    while (f"llma.layers.{num_layers}.attention_norm.weight"
           in merged_state_dict):
        num_layers += 1
    if model_type == 'mixtral':
        num_experts = 0
        while (f"llma.layers.0.feed_forward.experts.{num_experts}.w1.weight"
                in merged_state_dict):
            num_experts += 1
    else:
        num_experts = None
    hf_ckpts = []
    if model_type == 'llama':
        if "llma.rope.freqs" in merged_state_dict:
            del merged_state_dict["llma.rope.freqs"]
    for i in range(num_layers):
        hf_ckpt_shard = {}
        if model_type == 'llama':
            src_dst_name_mapping = [
                ("attention.wq.weight", "self_attn.q_proj.weight"),
                ("attention.wk.weight", "self_attn.k_proj.weight"),
                ("attention.wv.weight", "self_attn.v_proj.weight"),
                ("attention.wo.weight", "self_attn.o_proj.weight"),
                ("feed_forward.w3.weight", "mlp.up_proj.weight"),
                ("feed_forward.w2.weight", "mlp.down_proj.weight"),
                ("feed_forward.w1.weight", "mlp.gate_proj.weight"),
                ("attention_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
            ]
        elif model_type == 'mixtral':
            src_dst_name_mapping = [
                ("attention.wq.weight", "self_attn.q_proj.weight"),
                ("attention.wk.weight", "self_attn.k_proj.weight"),
                ("attention.wv.weight", "self_attn.v_proj.weight"),
                ("attention.wo.weight", "self_attn.o_proj.weight"),
                ("attention_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
                ("feed_forward.gate.weight", "block_sparse_moe.gate.weight"),
            ] + sum([[
                (f"feed_forward.experts.{exp_no}.w1.weight", f"block_sparse_moe.experts.{exp_no}.w1.weight"),
                (f"feed_forward.experts.{exp_no}.w2.weight", f"block_sparse_moe.experts.{exp_no}.w2.weight"),
                (f"feed_forward.experts.{exp_no}.w3.weight", f"block_sparse_moe.experts.{exp_no}.w3.weight"),
            ] for exp_no in range(num_experts)], start=[])
        else:
            raise NotImplementedError(f"Unsupported model type {model_type}")

        for src_key, dst_key in src_dst_name_mapping:
            dst_key = f"model.layers.{i}." + dst_key
            src_key = f"llma.layers.{i}." + src_key
            value = merged_state_dict[src_key]
            if "q_proj" in dst_key or "k_proj" in dst_key:
                # to be compatible with HuggingFace's pos embed implementation.
                if "q_proj" in dst_key:
                    n_heads = params["n_heads"]
                else:  # "k_proj" in dst_key:
                    n_heads = params.get("n_kv_heads", params["n_heads"])
                head_dim = value.size(0) // n_heads
                in_dim = value.size(1)
                value = value.view(
                    n_heads, head_dim // 2, 2, in_dim,
                ).transpose(1, 2).flatten(0, 2)
            hf_ckpt_shard[dst_key] = value
            del merged_state_dict[src_key]
        hf_ckpts.append(hf_ckpt_shard)

    hf_ckpts.append({})
    for src_key, dst_key in [
        ("llma.norm.weight", "model.norm.weight"),
        ("llma.output.weight", "lm_head.weight"),
        ("llma.tok_embeddings.weight", "model.embed_tokens.weight"),
    ]:
        hf_ckpts[-1][dst_key] = merged_state_dict[src_key]
        del merged_state_dict[src_key]
    assert len(merged_state_dict) == 0, (
        "Unknown key(s) in the source state dict: "
        + ", ".join(merged_state_dict.keys())
    )

    return hf_ckpts


def write_model_weights(hf_state_dict: List[Dict[str, torch.Tensor]], dest_dir: str) -> None:
    model_index = {
        "metadata": {"total_size": 0},
        "weight_map": {},
    }
    for shard_id, shard_state_dict in enumerate(hf_state_dict):
        shard_fn = (
            f"pytorch_model-{shard_id + 1:05d}-of-"
            f"{len(hf_state_dict):05d}.bin"
        )
        print(f"Writing to {shard_fn} ...")
        for key, value in shard_state_dict.items():
            model_index["weight_map"][key] = shard_fn
            model_index["metadata"]["total_size"] += (
                value.numel() * torch.finfo(value.dtype).bits
                * (2 if value.is_complex() else 1)
            )
        torch.save(shard_state_dict, os.path.join(dest_dir, shard_fn))
    with open(
        os.path.join(dest_dir, "pytorch_model.bin.index.json"), "w"
    ) as f:
        json.dump(model_index, f, indent=2)


def write_tokenizer(tokenizer_path: str, dest_dir: str) -> Any:
    try:
        from transformers import LlamaTokenizerFast
    except ImportError:
        print(
            "WARNING! The converted tokenizer will be the `slow` tokenizer. "
            "To use the fast, update your `tokenizers` library and re-run the "
            "tokenizer conversion."
        )
        LlamaTokenizerFast = None
    from transformers import LlamaTokenizer
    tokenizer_class = LlamaTokenizerFast or LlamaTokenizer
    tokenizer = tokenizer_class(tokenizer_path)
    tokenizer.save_pretrained(dest_dir)
    return tokenizer


def write_configs(
    params: Dict[str, Any],
    dtype: torch.dtype,
    dest_dir: str,
    vocab_size: int,
    model_type: str
) -> None:
    def calculate_hidden_dim():
        hidden_dim = params["dim"] * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if "ffn_dim_multiplier" in params:
            hidden_dim = int(hidden_dim * params["ffn_dim_multiplier"])
        multiple_of = params["multiple_of"]
        hidden_dim = (
            multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        )
        return hidden_dim

    if model_type == 'llama':
        config = {
            "architectures": [
                "LlamaForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": params["dim"],
            "initializer_range": 0.02,
            "intermediate_size": calculate_hidden_dim(),
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": params["n_heads"],
            "num_hidden_layers": params["n_layers"],
            "num_key_value_heads": params.get("n_kv_heads", params["n_heads"]),
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": params.get("norm_eps", 1e-5),
            "rope_theta": params.get("rope_theta", 10000),
            "rope_scaling": None if "rope_scaling" not in params else {
                "type": "linear",
                "factor": params["rope_scaling"],
            },
            "tie_word_embeddings": False,
            "torch_dtype": {
                torch.float16: "float16",
                torch.bfloat16: "bfloat16",
                torch.float32: "float32",
            }[dtype],
            "transformers_version": transformers.__version__,
            "use_cache": True,
            "vocab_size": vocab_size
        }
    elif model_type == 'mixtral':
        config = {
            "architectures": [
                "MixtralForCausalLM"
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": params["dim"],
            "initializer_range": 0.02,
            "intermediate_size": params["hidden_dim"],
            "max_position_embeddings": 32768,
            "model_type": "mixtral",
            "num_attention_heads": params["n_heads"],
            "num_experts_per_tok": 2,
            "num_hidden_layers": params["n_layers"],
            "num_key_value_heads": params.get("n_kv_heads", params["n_heads"]),
            "output_router_logits": False,
            "rms_norm_eps": params.get("norm_eps", 1e-5),
            "rope_theta": params.get("rope_theta", 10000),
            "rope_scaling": None if "rope_scaling" not in params or params["rope_scaling"] is None else {
                "type": "linear",
                "factor": params["rope_scaling"],
            },
            "router_aux_loss_coef": 0.02,
            "sliding_window": None,
            "tie_word_embeddings": False,
            "torch_dtype": {
                torch.float16: "float16",
                torch.bfloat16: "bfloat16",
                torch.float32: "float32",
            }[dtype],
            "transformers_version": transformers.__version__,
            "use_cache": True,
            "vocab_size": vocab_size
        }
    with open(os.path.join(dest_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if model_type == 'llama':
        generation_config = {
            "_from_model_config": True,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "transformers_version": transformers.__version__,
        }
    elif model_type == 'mixtral':
        generation_config = {
            "_from_model_config": True,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "transformers_version": transformers.__version__,
        }
    with open(os.path.join(dest_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)


def write_hf_ckpt(
    hf_state_dict: List[Dict[str, torch.Tensor]], dest_dir: str,
    tokenizer_path: str, params: Dict[str, Any], torch_dtype: torch.dtype,
    model_type: str = 'llama'
) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    print("Writing model weights ...")
    write_model_weights(hf_state_dict, dest_dir)
    print("Writing tokenizer ...")
    tokenizer = write_tokenizer(tokenizer_path, dest_dir)
    print("Writing configs ...")
    write_configs(params, torch_dtype, dest_dir, tokenizer.vocab_size, model_type)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def main() -> None:
    args = parse_arguments()

    src_config_file = f'{args.input_dir}/config.json'
    src_tokenizer_file = f'{args.input_dir}/tokenizer.model'
    src_weights_file = f'{args.input_dir}/consolidated.00.pth'
    dst_weights_path = f'{args.output_dir}'
    dtype = "bf16"
    mixtral = True

    params = {}
    with open(src_config_file) as f:
        params.update(json.load(f))

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]

    src_ckpt = torch.load(src_weights_file, map_location="cpu")
    print("#"*5," load OK")
    hf_ckpt = convert_merged_ckpt_to_hf(src_ckpt, params, 'mixtral' if mixtral else 'llama')
    print("#"*5," convert OK")
    write_hf_ckpt(
        hf_ckpt,
        dst_weights_path,
        src_tokenizer_file,
        params,
        torch_dtype,
        'mixtral' if mixtral else 'llama'
    )
    print("#"*5," write OK")


if __name__ == "__main__":
    main()
