## merge
python merge.py \
--config ./config/llama2_7B_sample.yaml \
--output_path ./merged_7B_4exp


python merge.py \
--config ./config/llama2_7B_sample_1exp.yaml \
--output_path ./merged_7B_1exp


python merge.py \
--config ./config/llama2_7B_sample_2exp.yaml \
--output_path ./merged_7B_2exp

python merge.py \
--config ./config/llama2_7B_sample_3exp.yaml \
--output_path ./merged_7B_3exp


python merge.py \
--config ./config/tanuki_dentaku.yaml \
--output_path ./merged_tanuki_dentaku



n_router_weights 384

## train
torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--repo_id ./merged_7B_4exp \
--wandb_project llama2_lora_ft_sample \
--wandb_entity if_local \
--include_lm_head \
--output_dir ./trained_7B_4exp


torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--repo_id ./merged_7B_1exp \
--wandb_project llama2_lora_ft_sample \
--wandb_entity if_local \
--include_lm_head \
--output_dir ./trained_7B_1exp

torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--repo_id ./merged_7B_2exp \
--wandb_project llama2_lora_ft_sample \
--wandb_entity if_local \
--include_lm_head \
--output_dir ./trained_7B_2exp

torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--repo_id ./merged_7B_3exp \
--wandb_project llama2_lora_ft_sample \
--wandb_entity if_local \
--include_lm_head \
--output_dir ./trained_7B_3exp

torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--repo_id ./merged_7B_4exp \
--wandb_project llama2_lora_ft_sample \
--wandb_entity if_local \
--include_lm_head \
--output_dir ./sample_trained


torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--tokenizer hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160 \
--repo_id ./merged_tanuki_dentaku \
--wandb_project router_tanuki_dentaku \
--wandb_entity weblab-geniac3 \
--output_dir /storage6/aa_fujimoto/router_tanuki



python upload_gpt2_to_hf_hub.py \
--repo_id team-sanai/llama2_7B_pretrain_2nd \
--tokenizer_dir team-sanai/unigram_32000 \
--test_prompt_text こんにちは \
--model_dir /storage6/aa_fujimoto/hf_llama2_7B
