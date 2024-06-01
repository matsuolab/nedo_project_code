import wandb
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict

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
BATCH_SIZE=32

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--wandb", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args

ratios = {
    "seed_10G": 0.2,
    "stack": 0.4,
    "s2": 0.3,
    "wiki": 0.1,
}
def make_dataset(tokenizer):
    MAX_LINES = 10000
    target_list = {
        "seed_10G": "team-sanai/seed_10G",
        "stack": "team-sanai/stack",
        "s2": "team-sanai/s2",
        "wiki": "team-sanai/wiki"
    }
    datasets = {name: load_dataset(path, split="train", num_proc=8) for name, path in target_list.items()}
    ds = []
    for name, dataset in datasets.items():
        lines = int(ratios[name]*MAX_LINES)
        ds_part = dataset.shuffle(seed=42).select(range(lines))
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

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.repo_id)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Attention層のパラメーターをfreeze
    #for name, param in model.named_parameters():
    #    if 'attn' in name:  # Attention層を識別
    #        param.requires_grad = False
       
    dataset = make_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        callbacks=[TokenCountCallback(max_token_count=MAX_TOKENS)]
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()