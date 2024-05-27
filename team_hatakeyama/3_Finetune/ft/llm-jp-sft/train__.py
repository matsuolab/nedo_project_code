import logging
from dataclasses import dataclass
from typing import Optional
import torch
from datasets import disable_caching, load_dataset, concatenate_datasets

from mergoo.models.modeling_llama import LlamaForCausalLM
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
    peft_target_model: Optional[str] = "llama-all"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            logger.warning(
                f"you should se the peft_target_modules when using peft_target_model"
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
    datasets = []
    for data_file in data_files:
        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]
        dataset = dataset.select_columns("text")
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()


    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    logger.info(training_args)
    logger.info(sft_training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info("Loading data")

    train_dataset = load_datasets(sft_training_args.data_files)
    if sft_training_args.eval_data_files:
        print("do eval")
        eval_dataset = load_datasets(sft_training_args.eval_data_files)
        training_args.do_eval = True
        training_args.evaluation_strategy = "epoch"
    else:
        eval_dataset = None

    logger.info("Formatting prompts")
    response_ids = tokenizer.encode(
        sft_training_args.response_template, add_special_tokens=False)[1:]
    if sft_training_args.instruction_template:
        instruction_ids = tokenizer.encode(
            sft_training_args.instruction_template, add_special_tokens=False)[1:]
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
    if sft_training_args.model_name_or_path.find("mergoo")>=0:
        #aceclerateをつかったマルチgpuでの学習がうまくいかなかった
        print("init mergoo model")

        model = LlamaForCausalLM.from_pretrained(
            sft_training_args.model_name_or_path, 
            device_map="auto",  #accelerateの場合はoffにする
            #**kwargs,
        )# 'gate' / router layers are untrained hence loaded warning would appeare for them

        # train only router (gating) layers
        n_weights, n_router_weights  = 0,0
        for name, weight in model.named_parameters():
            if "gate" not in name:
                weight.requires_grad_(False)
                n_router_weights += 1
            n_weights += 1
        print("train params:")
        print(n_weights)
        print(n_router_weights)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            sft_training_args.model_name_or_path,
            trust_remote_code=True,
            #device_map="auto",
        #device_map={'':device_string},
            **kwargs,
        )


    if training_args.gradient_checkpointing:
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        #peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length,
        neftune_noise_alpha=5,  # NEFTune https://qiita.com/m__k/items/23ced0db6846e97d41cd
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(
        #level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
