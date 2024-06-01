## train


expert wiki
```
torchrun --nproc_per_node=8 expert_train.py \
--repo_id team-sanai/llama2_7B_pretrain \
--wandb_project team_zoo_llama7b_wiki_expert \
--wandb_entity weblab-geniac3 \
--output_dir /storage6/aa_fujimoto/expert_wiki_llama2_7B \
--resume
```

expert math
```
torchrun --nproc_per_node=8 expert_train.py \
--repo_id team-sanai/llama2_7B_pretrain \
--wandb_project team_zoo_llama7b_math_expert \
--wandb_entity weblab-geniac3 \
--output_dir /storage6/aa_fujimoto/expert_math_llama2_7B \
--resume
```

expert novel
```
torchrun --nproc_per_node=8 expert_train.py \
--repo_id team-sanai/llama2_7B_pretrain \
--wandb_project team_zoo_llama7b_novel_expert \
--wandb_entity weblab-geniac3 \
--output_dir /storage6/aa_fujimoto/expert_novel_llama2_7B \
--resume
```




torchrun --nproc_per_node=8 expert_train.py \
--repo_id team-sanai/llama2_7B_pretrain \
--wandb_project llama2_lora_ft_sample \
--wandb_entity if_local \
--output_dir ./sample

## infarence
python infarence.py \
--peft_id="/storage6/aa_fujimoto/expert_wiki_llama2_7B/" \
--prompt="私の誕生日は"

python infarence.py \
--peft_id="/storage6/aa_fujimoto/expert_math_llama2_7B/" \
--prompt="私の誕生日は"

python infarence.py \
--peft_id="/storage6/aa_fujimoto/expert_novel_llama2_7B/" \
--prompt="私の誕生日は"

python infarence.py \
--prompt="私の誕生日は"

python infarence.py \
--repo_id="hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160" \
--tokenizer="hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160" \
--prompt="私の誕生日は"

python infarence.py \
--repo_id="hatakeyama-llm-team/tanuki_inst_0515test" \
--tokenizer="hatakeyama-llm-team/tanuki_inst_0515test" \
--prompt="""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
富士山の高さを教えてください

### 応答:"""





### llama2B_0.1B_lora_train.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geniac-team-sannai/ucllm_nedo_prod_MoE/blob/lora_sft/train/scripts/lora_sft/llama2B_0.1B_lora_train.ipynb)

### llama2B_0.1B_lora_mergoo.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geniac-team-sannai/ucllm_nedo_prod_MoE/blob/lora_sft/train/scripts/lora_sft/llama2B_0.1B_lora_mergoo.ipynb)

