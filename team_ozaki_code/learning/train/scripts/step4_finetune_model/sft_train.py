import logging
from dataclasses import dataclass
from typing import Optional

import yaml

import torch
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

disable_caching()

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: list[str]
    response_template: str
    eval_data_files: Optional[list[str]] = None
    instruction_template: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tag: Optional[str] = None

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs

def load_datasets(data_files):
    with open(data_files[0]) as file:
        data_files = yaml.safe_load(file)['datasets']
    datasets = []
    for data_file in data_files:
        #dataset = load_dataset("json", data_files=data_file)
        dataset = load_dataset(data_file['name'], split=data_file['split'])
        #dataset = dataset[data_file['split']]
        
        def create_propmt(example): 
            prompt = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{example[data_file['instruction']]}"
            if example[data_file['input']] is not None:
            #if data_file['input'] in example and len(example[data_file['input']]) > 0: 
                prompt += f"\n\n### 入力:\n{example[data_file['input']]}"
            prompt += f"\n\n### 応答:\n{example[data_file['output']]}"
            return prompt

        dataset = dataset.map(lambda example: {"text": create_propmt(example)})
        dataset = dataset.select_columns("text")
        logger.info(f"{data_file['name']}: {len(dataset)} records")
        logger.info(f"example: {dataset[:5]}")
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    logger.info("Loading data")

    train_dataset = load_datasets(sft_training_args.data_files)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files)
        training_args.do_eval = True
    else:
        eval_dataset = None

    logger.info("Formatting prompts")
    sft_training_args.response_template = sft_training_args.response_template + "\n" # sft_training_args.response_templateで\nを渡すと意図した文字にならない
    response_ids = tokenizer.encode(sft_training_args.response_template, add_special_tokens=False)[1:]
    if sft_training_args.instruction_template:
        instruction_ids = tokenizer.encode(sft_training_args.instruction_template, add_special_tokens=False)[1:]
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_ids, response_template=response_ids, tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_ids, tokenizer=tokenizer
        )

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )

    logger.info(model)

    peft_config: Optional[LoraConfig] = None
    if sft_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    # Initialize W&B for each node.
    #
    # To log to W&B for each node, we need to manually call `wandb.init()` with our customized parameters for each node
    # before `Trainer` class is initialized because `Trainer` class internally calls `wandb.init()` with the default parameters.
    # See: https://docs.wandb.ai/guides/integrations/huggingface#customize-wandbinit
    #
    # However, even though we manually call `wandb.init()` for each node, the training metrics (e.g. loss) will be logged only for the parent node
    # because `Trainer` class uses `WandbCallback` class and `WandbCallback` class internally calls `wandb.log()` only for the parent node.
    # See: https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L807-L809
    # But that's no problem because what we want is the system metrics (e.g. GPU utilization) and it will be successfully logged for each node.
    if sft_training_args.use_wandb:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        is_local_rank_0 = (torch.distributed.get_rank() % get_accelerator().device_count() == 0) if torch.distributed.is_initialized() else True
        if is_local_rank_0:
            import socket
            import wandb
            logger.info("Setting up wandb")
            wandb.init(entity=sft_training_args.wandb_entity, project=sft_training_args.wandb_project,
                       group=sft_training_args.wandb_group, name=socket.gethostname(),
                       tags=[sft_training_args.wandb_tag] if sft_training_args.wandb_tag else None)

    training_args.save_strategy="epoch"

    # text = train_dataset[0]['text']
    # text_ids = [tokenizer.encode(text)]
    # tokens = [tokenizer.tokenize(text)]
    # batch = collator(text_ids)
    # print(text)
    # print(text_ids)
    # print(tokens)
    # print(len(text_ids[0]))
    # print(batch)
    # print(sft_training_args.response_template, response_ids, tokenizer.tokenize(sft_training_args.response_template))

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
