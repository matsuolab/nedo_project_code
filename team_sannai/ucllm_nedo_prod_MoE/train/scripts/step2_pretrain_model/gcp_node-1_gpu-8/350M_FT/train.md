### set1
`/persistentshare/storage/team_sannai/team_la/seed/dataset1`のデータセットを使います

```
cd ucllm_nedo_prod_MoE/train/scripts/step2_pretrain_model/gcp_node-1_gpu-8/350M_FT
```


## (ii)*(ii)
wandbのproject名:llama2_350M_ii_ii
### s2
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 4 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/ii_ii/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/

```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 4 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/ii_ii/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```

### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 4 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/ii_ii/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```


## (o) attn, MLP freeze

wandbのproject名:llama2_350M_o_attn_MLP_freeze

### s2
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 4 \
--train_tokens_rate 1.94 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/attn_mlp_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 4 \
--train_tokens_rate 1.94 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/attn_mlp_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```

### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 4 \
--train_tokens_rate 1.94 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/attn_mlp_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```


## (o) attn freeze
wandbのproject名:llama2_350M_o_attn_freeze

### s2
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 4 \
--train_tokens_rate 1.35 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/attn_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 4 \
--train_tokens_rate 1.35 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/attn_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```

### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 4 \
--train_tokens_rate 1.35 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/attn_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```


## (o) MLP freeze
wandbのproject名:llama2_350M_o_MLP_freeze

### s2

```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 4 \
--train_tokens_rate 1.29 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/mlp_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 4 \
--train_tokens_rate 1.29 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/mlp_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```


### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 4 \
--train_tokens_rate 1.29 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/o/mlp_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/350M_llama2/checkpoint/gpt_0.35B_tok1B_lr3.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs16_g2_z1_pp1_seed1234_rebase/global_step60000/
```
