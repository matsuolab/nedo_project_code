### set1
`/persistentshare/storage/team_sannai/team_la/seed/dataset1`のデータセットを使います


## (ii)*(ii)
### s2
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/ii_ii/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### stack
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/ii_ii/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### wiki
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/ii_ii/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```


## (o) attn, MLP freeze
### s2
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--train_tokens_rate 1.94 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/attn_mlp_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### stack
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--train_tokens_rate 1.94 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/attn_mlp_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### wiki
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--train_tokens_rate 1.94 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/attn_mlp_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```


## (o) attn freeze
### s2
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--train_tokens_rate 1.35 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/attn_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### stack
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--train_tokens_rate 1.35 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/attn_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### wiki
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--train_tokens_rate 1.35 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/attn_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```


## (o) MLP freeze
### s2

```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--train_tokens_rate 1.29 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/mlp_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```

### stack
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--train_tokens_rate 1.29 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/mlp_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```


### wiki
```
time bash zero-0_dp-1_pp-1_tp-1_flashattn2-on.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--train_tokens_rate 1.29 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/o/mlp_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_gpt_v1/seed/gpt2_seed_multi_1B_w2000/checkpoint/gpt_0.125B_tok1B_lr6.0e-4_min1.0e-6_w30M_d1B_cosine_gbs32_mbs8_g4_z1_pp1_seed1234_rebase/global_step28000/
```
