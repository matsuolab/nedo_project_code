"""
!pip install git+https://github.com/if001/mergoo.git
"""

import argparse
import yaml
from dataclasses import dataclass, asdict, field

import torch
from mergoo.compose_experts import ComposeExperts

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--repo_id", type=str)
    # parser.add_argument("--wandb", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args

@dataclass
class ExpertConfig:
    expert_name: str
    model_id: str

@dataclass
class Config:
    model_type: str
    num_experts_per_tok: int
    base_model: list
    tokenizer: str = "team-sanai/unigram_32000"
    virtual_expert: bool = False
    experts: list[ExpertConfig] = field(default_factory=list)

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        return cls(**data)

def main():
    args = parse_arguments()
    config = Config.load(args.config_path)
    config_dict = asdict(config)

    expertmerger = ComposeExperts(config_dict, torch_dtype=torch.bfloat16)
    expertmerger.compose()
    expertmerger.save_checkpoint(args.output_path, config.tokenizer)

    # tokenizer = AutoTokenizer.from_pretrained("./"+model_id)
    # model = AutoModelForCausalLM.from_pretrained("./"+model_id)
    # tokenizer.push_to_hub(model_id)
    #model.push_to_hub(model_id)

if __name__ == "__main__":
    main()