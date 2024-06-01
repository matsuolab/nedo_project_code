from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload_repo_id", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="team-sanai/llama2_7B_pretrain")
    parser.add_argument("--peft_id", type=str, required=True)

    args = parser.parse_args()
    print("args: ", args)
    return args

def main():
    args = parse_arguments()
    base_model = AutoModelForCausalLM.from_pretrained(args.repo_id)
    model = PeftModel.from_pretrained(base_model, model_id=args.peft_id)
    model.push_to_hub(args.upload_repo_id)

if __name__ == "__main__":
    main()