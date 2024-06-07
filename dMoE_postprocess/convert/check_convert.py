#
# author : 加藤　純
#
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


input_dir = sys.argv[1]
output_dir = sys.argv[2]
n_experts = int(sys.argv[3])
n_layers = int(sys.argv[4])
n_heads = int(sys.argv[5])
n_kv_heads = int(sys.argv[6])

tfm_src_dst_mapping = [
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
            ] for exp_no in range(n_experts)], start=[])

emb_ln_src_dst_mapping = [
    ("tok_embeddings.weight", "model.embed_tokens.weight"),
    ("norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
]


old_sd = torch.load(input_dir + "/consolidated.00.pth", map_location="cpu")
new_model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype="auto")
new_sd = {k : v for k, v in new_model.named_parameters()}


for i in range(n_layers):
    for src_key, dst_key in tfm_src_dst_mapping:
        src_value = old_sd[f"layers.{i}." + src_key]
        dst_value = new_sd[f"model.layers.{i}." + dst_key]
        if dst_key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight"]:
            if n_kv_heads > 0 and dst_key == "self_attn.k_proj.weight":
                n_heads = n_kv_heads
            head_dim = dst_value.size(0) // n_heads
            in_dim = dst_value.size(1)
            dst_value = dst_value.view(n_heads, 2, head_dim // 2, in_dim,).transpose(1, 2).flatten(0, 2)

        assert torch.equal(src_value, dst_value), f'mismatch layers.{i}.{src_key}'

for src_key, dst_key in emb_ln_src_dst_mapping:
    assert torch.equal(old_sd[src_key], new_sd[dst_key]), f'mismatch layers.{i}.{src_key}'
