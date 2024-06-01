### set1
`/persistentshare/storage/team_sannai/team_la/seed/dataset1`のデータセットを使います


## (ii)*(ii)
### s2
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/ii_ii/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/ii_ii/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/ii_ii/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```


## (o) attn, MLP freeze
### s2
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--train_tokens_rate 4.20 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/attn_mlp_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--train_tokens_rate 4.20 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/attn_mlp_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--train_tokens_rate 4.20 \
--freeze_attn true \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/attn_mlp_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```


## (o) attn freeze
### s2
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--train_tokens_rate 1.12 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/attn_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--train_tokens_rate 1.12 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/attn_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--train_tokens_rate 1.12 \
--freeze_attn true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/attn_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```


## (o) MLP freeze
### s2

```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type s2 \
--train_tokens_in_million 2 \
--train_tokens_rate 2.22 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/mlp_freeze/s2 \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```

### stack
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type stack \
--train_tokens_in_million 2 \
--train_tokens_rate 2.22 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/mlp_freeze/stack \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```


### wiki
```
time bash llama2.sh \
--input_tokenizer_file "mistralai/Mixtral-8x7B-v0.1" \
--dataset_index 1 \
--dataset_type wiki \
--train_tokens_in_million 2 \
--train_tokens_rate 2.22 \
--freeze_head true \
--master_port 6000 \
--output_model_dir /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/o/mlp_freeze/wiki \
--checkpoint_path /persistentshare/storage/team_sannai/team_la/model_weights/0.1B_llama2/seed/checkpoint/global_step40000/
```
