## train

```
python train.py \
--repo_id team-sanai/gpt2_seed_0.1B_ii_merged \
--output_dir ./output_ii
```

```
python train.py \
--repo_id team-sanai/gpt2_seed_0.1B_o_attn_mlp_freeze_merge \
--output_dir ./output_attn_mlp_freeze
```

```
python train.py \
--repo_id team-sanai/gpt2_seed_0.1B_o_attn_freeze_merge \
--output_dir ./output_attn_freeze
```

```
python train.py \
--repo_id team-sanai/gpt2_seed_0.1B_o_mlp_freeze_merge \
--output_dir ./output_mlp_freeze
```




## llama2
```
python train.py \
--repo_id team-sanai/llama2_0.1B_ii_merged \
--wandb llama2_0-1B_router \
--output_dir ./output_ii
```

```
python train.py \
--repo_id team-sanai/llama2_0.1B_attn_mlp_freeze_merged \
--wandb llama2_0-1B_router \
--output_dir ./output_attn_mlp_freeze
```

```
python train.py \
--repo_id team-sanai/llama2_0.1B_attn_freeze_merged \
--wandb llama2_0-1B_router \
--output_dir ./output_attn_freeze
```

```
python train.py \
--repo_id team-sanai/llama2_0.1B_mlp_freeze_merged \
--wandb llama2_0-1B_router \
--output_dir ./output_mlp_freeze
```



python train.py \
--repo_id team-sanai/llama2_0.1B_seed \
--wandb llama2_0-1B_router \
--output_dir ./gomi