## train

```
python train.py \
--repo_id team-sanai/llama2_seed_350M_ii_merged \
--output_dir ./output_ii
```

```
python train.py \
--repo_id team-sanai/llama2_350M_o_attn_mlp_freeze_merge \
--output_dir ./output_attn_mlp_freeze
```

```
python train.py \
--repo_id team-sanai/llama2_350M_o_attn_freeze_merge \
--output_dir ./output_attn_freeze
```

```
python train.py \
--repo_id team-sanai/llama2_350M_o_mlp_freeze_merge \
--output_dir ./output_mlp_freeze
```

