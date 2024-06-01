python ~/tmp_scripts/llama_checkpoint_conversion.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "/persistentshare/storage/team_ozaki/personal/shioya/megatron_checkpoint/gpt_0.125B_tok2B_ja_wiki_train_0_14_lr6.0e-4_min1.0e-6_w20M_d2B_cosine_gbs256_mbs4_g_pp1_seed1234_rebase_llama2_except_for_kv_head" \
--save_path "/persistentshare/storage/team_ozaki/personal/shioya/hf_checkpoint" \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
#--megatron-path "PATH_TO_MEGATRON_SOURCE_CODE"