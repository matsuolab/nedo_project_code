import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_and_model_dir", type=str, required=True)
    parser.add_argument("--output_model_name", type=str, required=True)
    parser.add_argument("--test_prompt_text", type=str, default="Once upon a time,")
    #parser.add_argument("--instruction_template", type=str, default="### Human:")
    #parser.add_argument("--response_template", type=str, default="### Assistant:")
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir, use_fast=False)
    print(input_tokenizer_and_model_dir)
    if "adapter_config.json" in os.listdir(input_tokenizer_and_model_dir):
        peft_config = PeftConfig.from_pretrained(input_tokenizer_and_model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            return_dict=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, input_tokenizer_and_model_dir)
        model = model.merge_and_unload()
        print('adapter merged')
    else:    
        model = AutoModelForCausalLM.from_pretrained(input_tokenizer_and_model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    # 各パラメータのデータ型を出力
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    return tokenizer, model


def test_tokenizer_and_model(tokenizer, model, prompt_text: str) -> str:
    encoded_prompt_text = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        encoded_generation_text = model.generate(encoded_prompt_text, max_new_tokens=100)[0]
    decoded_generation_text = tokenizer.decode(encoded_generation_text)
    return decoded_generation_text


def main() -> None:
    args = parse_arguments()

    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(args.input_tokenizer_and_model_dir)
    test_prompt_text = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{args.test_prompt_text}\n\n### 応答:\n"
    local_decoded_generation_text = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt_text)

    # Checks the generated text briefly.
    print()
    print(f"{args.test_prompt_text = }")
    print(f"{local_decoded_generation_text = }")
    print()
    if len(local_decoded_generation_text) <= len(args.test_prompt_text):
        print("Error: The generated text should not be shorter than the prompt text."
              " Something went wrong, so please check either the local tokenizer or the local model."
              " This program will exit without uploading the tokenizer and the model to HuggingFace Hub.")
        return


    # Uploads the local tokenizer and the local model to HuggingFace Hub.
    local_tokenizer.push_to_hub(args.output_model_name)
    local_model.push_to_hub(args.output_model_name)

    # Loads and tests the remote tokenizer and the remote model.
    # huggingface_username = HfApi().whoami()["name"]
    # remote_tokenizer, remote_model = load_tokenizer_and_model(os.path.join(args.output_model_name))
    # test_prompt_text = f"{args.instruction_template} {args.test_prompt_text}{args.response_template}"
    # remote_decoded_generation_text = test_tokenizer_and_model(remote_tokenizer, remote_model, test_prompt_text)

    # # Checks the generated text briefly.
    # print()
    # print(f"{args.test_prompt_text = }")
    # print(f"{remote_decoded_generation_text = }")
    # print()
    # if len(remote_decoded_generation_text) <= len(args.test_prompt_text):
    #     print("Error: The generated text should not be shorter than the prompt text."
    #           " Something went wrong, so please check either the remote tokenizer or the remote model.")
    #     return


if __name__ == "__main__":
    main()
