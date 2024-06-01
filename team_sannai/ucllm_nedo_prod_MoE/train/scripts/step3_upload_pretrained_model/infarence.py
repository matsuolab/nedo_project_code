import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default="team-sanai/unigram_32000")
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--test_prompt_text", type=str, default="Once upon a time,")
    args = parser.parse_args()
    print(f"{args = }")
    return args

def test_tokenizer_and_model(tokenizer, model, prompt_text: str) -> str:
    encoded_prompt_text = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    print("encoded_prompt_text: ", encoded_prompt_text)
    with torch.no_grad():
        encoded_generation_text = model.generate(encoded_prompt_text, max_new_tokens=50)[0]
    print("encoded_generation_text: ", encoded_generation_text)
    decoded_generation_text = tokenizer.decode(encoded_generation_text)
    return decoded_generation_text


def main() -> None:
    args = parse_arguments()

    # Loads and tests the local tokenizer and the local model.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(args.repo_id, device_map="auto")
    local_decoded_generation_text = test_tokenizer_and_model(tokenizer, model, args.test_prompt_text)

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
    #tokenizer.push_to_hub(args.output_model_name)
    # model.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()