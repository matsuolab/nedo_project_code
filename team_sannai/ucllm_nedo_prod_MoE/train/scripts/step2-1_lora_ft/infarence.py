from peft import PeftModel
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="team-sanai/llama2_7B_pretrain")
    parser.add_argument("--peft_id", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="team-sanai/unigram_32000")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt_file", type=str)
    # parser.add_argument("--lora", type=str, required=True)
    # parser.add_argument("--load_8bit", action='store_true')

    args = parser.parse_args()
    print("args: ", args)
    return args

def load_prompt(prompt_file = None, prompt = None):
    if prompt:
        return [prompt]
    li = []
    with open(prompt_file) as f:
        li = f.readlines()
    return li

def main():
    args = parse_arguments()

    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id, 
        quantization_config=quantization_config
    )
    model.load_adapter(args.peft_id)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)

    prompt = load_prompt(prompt_file=args.prompt_file, prompt=args.prompt)

    for v in prompt:
        print("input: ", v)
        input_ids = tokenizer(v, return_tensors="pt", add_special_tokens=False).input_ids
        peft_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(do_sample=True, repetition_penalty=1.1))
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        print("output: ", peft_model_text_output)
        print("="*100)

if __name__ == "__main__":
    main()