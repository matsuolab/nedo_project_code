import wandb
import torch
import numpy as np
import argparse
import os
from typing import Optional

from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer,
    TrainingArguments, 
    DataCollatorForLanguageModeling,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from callbacks import ComputeThroughputCallback, TokenCountCallback


from expert_train import (
    parse_arguments,
    make_dataset,
    DDPTrainer
)

MAX_TOKENS = 2 * 1000 * 1000

MAX_LENGTH = 4096
BATCH_SIZE=1
LOGGING_STEPS=2
SAVE_STEPS=100
NUM_GPUS=int(torch.cuda.device_count())
LOCAL_RANK = int(os.environ['LOCAL_RANK'])

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

def rank_0_print(*args, **kwargs):
    if LOCAL_RANK == 0:
        print(*args, **kwargs)

def init_dist(rank=0, world_size=8):
    print("init distributed...")
    # os.environ['MASTER_ADDR'] = '172.16.0.9'
    # os.environ['MASTER_PORT'] = '6010'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.distributed.barrier()

def main():
    args = parse_arguments()
    init_dist(rank=LOCAL_RANK, world_size=NUM_GPUS)
    if LOCAL_RANK == 0:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.repo_id, torch_dtype=torch_dtype)

    target_modules=["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    if args.include_lm_head:
        print("include lm_head...")
        target_modules=["mlp.gate_proj","mlp.up_proj","mlp.down_proj", "lm_head"]
    peft_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1,
        # task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    if LOCAL_RANK == 0:
        model.print_trainable_parameters()

    device = torch.device(f"cuda:{dist.get_rank()}" if NUM_GPUS > 1 else "cuda")
    model.to(device)
    model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    # # Attention層のパラメーターをfreeze
    # for name, param in model.named_parameters():
    #     if 'attn' in name:  # Attention層を識別
    #         param.requires_grad = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("--- making dataset ... ---")
    dataset = make_dataset(tokenizer)
    rank_0_print(dataset)

    print("--- training start ... ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        seed=42,
        data_seed=42,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=BATCH_SIZE,
        warmup_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        weight_decay=0.01,
        logging_dir=args.output_dir,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        save_strategy="steps",
        learning_rate=6.0e-5,
        # min_lr
        save_total_limit=3,
        save_steps=SAVE_STEPS,
        report_to="wandb",
        bf16=True
    )    
    
    computeThroughput = ComputeThroughputCallback(
        vocab_size=model.module.config.vocab_size,
        seq_length=model.module.config.max_sequence_length,
        batch_size=BATCH_SIZE,
        num_layers=model.module.config.num_hidden_layers,
        hidden_size=model.module.config.hidden_size,
        world_size=1,
        log_steps=LOGGING_STEPS,
    )
    tokenCounter = TokenCountCallback(max_token_count=MAX_TOKENS)
    trainer = DDPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        callbacks=[tokenCounter, computeThroughput]
    )
    if trainer.is_world_process_zero():
        trainer.train()
        model.save_pretrained(args.output_dir, save_embedding_layers=args.include_lm_head)
        # torch.save(model.state_dict(), args.output_dir)

    if LOCAL_RANK == 0 and args.upload_repo_id:
        print("--- push to hf ---")
        # output_model_id = "team-sanai/llama2_0.1B_lora_sample"
        # model.push_to_hub(args.upload_repo_id, save_embedding_layers=True)
        model.push_to_hub(args.upload_repo_id)

if __name__ == "__main__":
    main()