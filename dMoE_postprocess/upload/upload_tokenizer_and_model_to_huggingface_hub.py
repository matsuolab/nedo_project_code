import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_and_model_dir", type=str, required=True)
    parser.add_argument("--output_model_name", type=str, required=True)
    parser.add_argument("--test_prompt_text", type=str, default="Once upon a time,")
    #ユーザではなく、組織のリポジトリにアップロード
    parser.add_argument("--use_orgs", action='store_true')
    #推論のみを実行し、ULは実施しない
    parser.add_argument("--no_upload", action='store_true')
    args = parser.parse_args()
    print(f"|--{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
    model = AutoModelForCausalLM.from_pretrained(input_tokenizer_and_model_dir, torch_dtype="auto")
    return tokenizer, model


def test_tokenizer_and_model(tokenizer, model, prompt_text: str) -> str:
    encoded_prompt_text = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        encoded_generation_text = model.generate(encoded_prompt_text, max_new_tokens=50)
    decoded_generation_text = tokenizer.decode(encoded_generation_text[0], skip_special_tokens=True)
    return decoded_generation_text


def main() -> None:
    args = parse_arguments()

    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(args.input_tokenizer_and_model_dir)
    local_decoded_generation_text = test_tokenizer_and_model(local_tokenizer, local_model, args.test_prompt_text)

    # Checks the generated text briefly.
    print(f'|--check : pytorch_model.bin')
    print(f"|--{args.test_prompt_text = }")
    print(f"|--{local_decoded_generation_text = }")
    print()
    if len(local_decoded_generation_text) <= len(args.test_prompt_text):
        print("Error: The generated text should not be shorter than the prompt text."
              " Something went wrong, so please check either the local tokenizer or the local model."
              " This program will exit without uploading the tokenizer and the model to HuggingFace Hub.")
        return

    if not args.no_upload:
        huggingface_workname =  HfApi().whoami()["orgs"][0]["name"] if args.use_orgs else HfApi().whoami()["name"]
        repository_name = os.path.join(huggingface_workname, args.output_model_name)

        # Uploads the local tokenizer and the local model to HuggingFace Hub.
        local_tokenizer.push_to_hub(repository_name)
        local_model.push_to_hub(repository_name)

        # Loads and tests the remote tokenizer and the remote model.
        remote_tokenizer, remote_model = load_tokenizer_and_model(repository_name)
        remote_decoded_generation_text = test_tokenizer_and_model(remote_tokenizer, remote_model, args.test_prompt_text)

        # Checks the generated text briefly.
        print(f'|--check : model-XX-of-YY.safetensors')
        print(f"|--{args.test_prompt_text = }")
        print(f"|--{remote_decoded_generation_text = }")
        print()
        if len(remote_decoded_generation_text) <= len(args.test_prompt_text):
            print("Error: The generated text should not be shorter than the prompt text."
                  " Something went wrong, so please check either the remote tokenizer or the remote model.")
            return


if __name__ == "__main__":
    main()
