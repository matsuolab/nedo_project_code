import json
import sys
sys.path.append("~/pretrain/models")
from deepseek.modeling_deepseek import DeepseekForCausalLM
from deepseek.configuration_deepseek import DeepseekConfig
import torch

model_dir = '../models/deepseek'
config_file = '../models/deepseek/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)
    config = DeepseekConfig.from_dict(config)

print("load model")
model = DeepseekForCausalLM(config).to(torch.bfloat16)
print("save model")
model.save_pretrained(model_dir)
print("done")
