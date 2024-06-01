import os
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

parser = argparse.ArgumentParser(description='test converted huggingface model')
parser.add_argument('model_dir', type=str, help='Path to the huggingface model')
args = parser.parse_args()



model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False,
    max_position_embeddings=8,
)

model.model.layers = model.model.layers[:8]

model.eval()
with torch.no_grad():
    token_ids = torch.tensor([[    1, 20811,   349,   396, 13126,   369, 13966,   264]])
    labels = torch.tensor([[20896, 26570, 20896, 21876, 25931, 25931, 20896, 20896]])
    attention_mask = torch.ones_like(token_ids)
    position_ids = attention_mask.long().cumsum(-1) - 1

    model_inputs = {
        'input_ids': token_ids.to(model.device),
        'labels': labels.to(model.device),
        'past_key_values': None,
        'use_cache': True,
        'position_ids': position_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device),
        'output_attentions': True,
        'output_hidden_states': True,
        'return_dict': True
    }
    outputs = model(**model_inputs)

    print(outputs.logits)
