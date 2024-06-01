import wandb
import torch
import numpy as np
import argparse
import os
from typing import Optional
os.environ["TOKENIZERS_PARALLELISM"]="false"

from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

print("torch version: ", torch.version.cuda)

from callbacks import ComputeThroughputCallback, TokenCountCallback
from prepare_dataset import prepare_dataset


MAX_TOKENS = 8 * 1000 * 1000 * 1000

MAX_LENGTH = 4096
# MAX_LENGTH=10
# BATCH_SIZE=4
# BATCH_SIZE=1024
BATCH_SIZE=1
# BATCH_SIZE=2
# BATCH_SIZE=4
GC_STEPS=1

LOGGING_STEPS=2
SAVE_STEPS=100
NUM_GPUS=int(os.environ['WORLD_SIZE'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])

if LOCAL_RANK == 0:
    import transformers
    transformers.logging.set_verbosity_info()


def rank_0_print(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--upload_repo_id", type=str)
    parser.add_argument("--tokenizer", type=str, default="team-sanai/unigram_32000")
    parser.add_argument("--include_lm_head", action='store_true')
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()
    print("args: ", args)
    return args

ratios = {
    "atlas": 1.0,
    "math_ins": 1.0,
    "basic": 1.0,
    "gsm8k": 1.0
}
#ratios = {
#    "news_0": 1.0,
#    "news_1": 1.0,
#    "news_2": 1.0,
#    "wiki_en": 1.0,
#    "wiki_ja": 1.0,
#}
ratios = {
    "aozora":1.0,
    "narou": 1.0
}
def make_dataset(tokenizer):
    target_list = {
        "atlas": "/storage6/corpus/category/MATH/raw/AtlasMathSets/AtlasMathSets_text_jsonl",
        "math_ins": "/storage6/corpus/category/MATH/raw/OpenMathInstruct/ja_math.jsonl",
        "basic": "/storage6/corpus/category/MATH/raw/basicMath/basicMath_10m_jsonl",
        "gsm8k": "/storage6/aa_fujimoto/for_expert_dataset/gsm8k.jsonl"
    }
    #target_list = {
    #    "news_0": "/storage6/corpus/category/NEWS/raw/JA/ja_news/news_00.jsonl",
    #    "news_1": "/storage6/corpus/category/NEWS/raw/JA/ja_news/news_01.jsonl",
    #    "news_2": "/storage6/corpus/category/NEWS/raw/JA/ja_news/news_02.jsonl",
    #    "wiki_en": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_en/merged_expert_en_wikipedia_4.0.jsonl",
    #    "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl"
    #}
    target_list = {
        "aozora":"/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
        "narou":"/storage6/dataset/pretrain/gen_experet/narou/merged_narou_ja_16.0.jsonl" 
    }
    datasets = {name: load_dataset("json", data_files=path, split="train", num_proc=8) for name, path in target_list.items()}
    ds = []
    # print(datasets)
    for name, dataset in datasets.items():
        rank_0_print(name, ratios[name])
        # ds_part = dataset.shuffle(seed=42).select(range(100))
        # ds_part = dataset.shuffle(seed=42)
        ds_part = dataset
        filtered_list = []
        for name in ds_part.column_names:
            if "text" != name:
                filtered_list.append(name)
        ds_part = ds_part.remove_columns(filtered_list)
        ds.append(ds_part)
    combined_dataset = concatenate_datasets(ds)
    rank_0_print("dataset", combined_dataset)
    return combined_dataset.shuffle(seed=42).train_test_split(test_size=0.1)


class DDPTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        sampler = self._get_train_sampler()
        data_loader = DataLoader(self.train_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=sampler is None,
                                 sampler=sampler,
                                 collate_fn=self.data_collator,
                                 drop_last=self.args.dataloader_drop_last,
                                 )
        # return self.accelerator.prepare(data_loader)
        return data_loader

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=NUM_GPUS, rank=dist.get_rank(), shuffle=True) if NUM_GPUS > 1 else None
        return train_sampler

def main():
    args = parse_arguments()
    # LOCAL_RANK = args.local_rank
    if LOCAL_RANK == 0:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            args.repo_id,
            # torch_dtype=torch.float16
            )
    
    target_modules=["gate_proj","up_proj","down_proj"]
    if args.include_lm_head:
        rank_0_print("include lm_head...")
        target_modules=["gate_proj","up_proj","down_proj","lm_head"]
    
    peft_config = LoraConfig(
        r=640, lora_alpha=32, lora_dropout=0.1,
        # task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # # Attention層のパラメーターをfreeze
    # for name, param in model.named_parameters():
    #     if 'attn' in name:  # Attention層を識別
    #         param.requires_grad = False
    if LOCAL_RANK == 0:
        for v in model.state_dict():
            print(v, model.state_dict()[v].shape)
        print("="*100)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    rank_0_print("--- making dataset ... ---")
    dataset = make_dataset(tokenizer)
    train_dataset = prepare_dataset(dataset["train"], tokenizer)
    test_dataset = prepare_dataset(dataset["test"], tokenizer)

    rank_0_print("--- training start ... ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,
        seed=42,
        data_seed=42,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GC_STEPS,
        warmup_steps=300,
        evaluation_strategy="steps",
        eval_steps=1000,
        weight_decay=0.01,
        optim="adamw_apex_fused",
        # optim="adafactor",
        logging_dir=args.output_dir,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        learning_rate=6.0e-5,
        # min_lr
        save_strategy="steps",
        save_total_limit=3,
        save_steps=SAVE_STEPS,
        report_to="wandb",
        # bf16=True,
        fp16=True,
        ddp_backend="nccl",
        # half_precision_backend="apex",
        deepspeed=args.ds_config_path,
        dataloader_pin_memory=True,
        dataloader_num_workers=16,
        # torch_compile=True,
        # num_workers=16,
        fsdp="full_shard",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
    )
    rank_0_print("parallel_mode: ", training_args.parallel_mode)
    rank_0_print("world_size", training_args.world_size)

    computeThroughput = ComputeThroughputCallback(
        vocab_size=model.config.vocab_size,
        #seq_length=model.config.max_sequence_length,
        seq_length=model.config.max_position_embeddings,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
        world_size=NUM_GPUS,
        log_steps=LOGGING_STEPS,
    )
    tokenCounter = TokenCountCallback(max_token_count=MAX_TOKENS)
    trainer = DDPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[computeThroughput]
    )

    trainer.train(resume_from_checkpoint=args.resume)
    rank_0_print("train done..")

    model.save_pretrained(args.output_dir,
            # save_embedding_layers=args.include_lm_head,
            is_main_process=LOCAL_RANK==0)
    rank_0_print("save...") 
    
    for v in model.state_dict():
        rank_0_print(v, model.state_dict()[v].shape)
    rank_0_print("="*100)

    if LOCAL_RANK == 0 and args.upload_repo_id:
        print("--- push to hf ---")
        # model.push_to_hub(args.upload_repo_id, save_embedding_layers=True)
        model.push_to_hub(args.upload_repo_id)
        print("upload done...") 
    if LOCAL_RANK == 0:
        wandb.finish()
if __name__ == "__main__":
    main()
