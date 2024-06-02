import os
import gc
import time
import pytz
import torch
import transformers
import wandb
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOTrainer
from dotenv import load_dotenv

import logging
from logging import getLogger

JST = pytz.timezone('Asia/Tokyo')

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
wb_token = os.getenv('WANDB_API_KEY')

wandb.login()
wandb.init()

model_name = ""
new_model = ""
dataset_name = ""

@dataclass
class TrainerArguments:
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_train_epochs: int = 3
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "linear"
    dataloader_num_workers: int = 4
    save_strategy: str = "steps"
    save_steps: int = 160
    save_total_limit: int = 6
    logging_steps: int = 1
    output_dir: str = new_model
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.03
    warmup_ratio: float = 0.1
    weight_decay: float = 0.001
    bf16: bool = True
    report_to: str = "wandb"
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    remove_unused_columns: bool = False

@dataclass
class DPOTrainerArguments:
    beta: float = 0.5
    max_prompt_length: int = 1024
    max_length: int = 1024
    loss_type: str = "sigmoid"

def create_prompt(example):
    prompt = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{example['instruction']}"
    if example['input'] is not None:
        prompt += f"\n\n### 入力:\n{example['input']}"
    prompt += "\n\n### 応答:\n"
    return prompt

def chatml_format(example):
    # Format instruction
    prompt = create_prompt(example)
    # Format chosen answer
    chosen = example['chosen'] + "\n"
    # Format rejected answer
    rejected = example['rejected'] + "\n"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    trainer_argument = TrainerArguments()
    dpo_argument = DPOTrainerArguments()

    dataset = load_dataset(dataset_name, split="train")
    # Save columns
    original_columns = dataset.column_names
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Format dataset
    dataset = dataset.map(
        chatml_format,
        remove_columns=original_columns
    )

    dataset = dataset.train_test_split(test_size=0.01)

    # Set torch dtype and attention implementation
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    # LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # Model to fine-tune
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation
   )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=new_model,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=trainer_argument.per_device_train_batch_size,
        gradient_accumulation_steps=trainer_argument.gradient_accumulation_steps,
        learning_rate=trainer_argument.learning_rate,
        warmup_ratio=trainer_argument.warmup_ratio,
        num_train_epochs=trainer_argument.num_train_epochs,
        gradient_checkpointing=trainer_argument.gradient_checkpointing,
        logging_steps=trainer_argument.logging_steps,
        save_steps=trainer_argument.save_steps,
        eval_steps=trainer_argument.eval_steps,
        save_total_limit=trainer_argument.save_total_limit,
        bf16=trainer_argument.bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        lr_scheduler_type="linear",
        report_to="wandb",
        optim=trainer_argument.optim,
        torch_compile=True,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=dpo_argument.max_prompt_length,
        max_length=dpo_argument.max_length,
        loss_type=dpo_argument.loss_type,
        beta=dpo_argument.beta
    )

    # Fine-tune model with DPO
    logger.info("Start Training.")
    dpo_trainer.train()

    # Save artifacts
    logger.info("Start Saving.")
    final_output_dir = os.path.join(new_model, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Flush memory
    del dpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    start = time.time()

    main()

    end = time.time()
    logger.info(f"実行時間:{end - start}s")