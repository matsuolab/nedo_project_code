# load the composed checkkpoint
import torch
from mergoo.models.modeling_mistral import MistralForCausalLM

import wandb
import torch
import numpy as np
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer,
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)

MAX_LENGTH = 512
MAX_TOKENS = 2 * 1000 * 1000
BATCH_SIZE=1

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
    parser.add_argument("--wandb", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args

ratios = {
    "wiki_ja": 1.0,
}
def make_dataset(tokenizer):
    target_list = {
        "wiki_ja": "izumi-lab/wikinews-ja-20230728",
    }
    datasets = {name: load_dataset(path, split="train", num_proc=8) for name, path in target_list.items()}
    ds = []
    print(datasets)
    for name, dataset in datasets.items():
        print(name, ratios[name])
        ds_part = dataset.select(range(10))
        # ds_part = dataset.shuffle(seed=42).select(range(10))
        filtered_list = []
        for name in ds_part.column_names:
            if "text" != name:
                filtered_list.append(name)
        ds_part = ds_part.remove_columns(filtered_list)
        ds.append(ds_part)
    combined_dataset = concatenate_datasets(ds)

    count = 0
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", max_length=MAX_LENGTH, truncation=True)
        nonlocal count
        for v in tokens["input_ids"]:
            count += len(v)
        return tokens
    
    tokenized_datasets = combined_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    print("total tokens: ", count)
    print("tokenized_datasets")
    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
    return tokenized_datasets

class TokenCountCallback(TrainerCallback):
    def __init__(self, max_token_count):
        self.max_token_count = max_token_count
        self.token_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.token_count += args.per_device_train_batch_size * args.gradient_accumulation_steps
        if self.token_count >= self.max_token_count:
            print("current tokens: ", self.token_count)
            print(f"指定されたトークン数 {self.max_token_count} に到達。学習を終了します。")
            control.should_training_stop = True

def main():
    args = parse_arguments()
    wandb.init(project=args.wandb)

    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("team-sanai/unigram_32000")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.repo_id)

    model = MistralForCausalLM.from_pretrained(
        "data/mistral_lora_moe",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # train only router (gating) layers
    n_weights, n_router_weights  = 0,0
    for name, weight in model.named_parameters():
        if "gate" in name:
            weight.requires_grad_(True)
            n_router_weights += 1
        else:
            weight.requires_grad_(False)
        n_weights += 1
    n_weights, n_router_weights


    print("--- making dataset ... ---")
    dataset = make_dataset(tokenizer)
    print(dataset)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        seed=42,
        data_seed=42,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=10,
        warmup_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        weight_decay=0.01,
        logging_dir=args.output_dir,
        save_strategy="steps",
        learning_rate=6.0e-5,
        # min_lr
        save_total_limit=3,
        save_steps=100,
        report_to="wandb"
    )
    print("--- training start ... ---")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        callbacks=[TokenCountCallback(max_token_count=MAX_TOKENS)]
    )

    trainer.train()
    model.save_pretrained(args.output_dir, save_embedding_layers=True)

    print("--- push to hf ---")
    output_model_id = "team-sanai/llama2_0.1B_lora_sample"
    model.push_to_hub(output_model_id, save_embedding_layers=True)


if __name__ == "__main__":
    main()