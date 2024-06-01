import argparse
import os
import re
import zipfile
from pprint import pprint

import torch

from transformers import AutoTokenizer, GPT2Config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str, help='Input DeepSpeed Checkpoint file')
    args = parser.parse_args()
    print(f'args = {args}')
    return args

args = parse_arguments()
input_state_dict = torch.load(args.input_file)

ds_args = input_state_dict.get("args", None)
print(ds_args)


# The model.
model = input_state_dict["model"]
print(model.keys())
# The language model.
lm = model["language_model"]
# The embeddings.
embeddings = lm["embedding"]
# The encoder.
encoder = lm['encoder']

for emb_type in embeddings.keys():
    for name, params in embeddings[emb_type].items():
        print(emb_type, name, params.size(), params.dtype)


for name, params in encoder.items():
    print(name, params.size(), params.dtype)

if 'word_embeddings_for_head' in model.keys():
    head = model['word_embeddings_for_head']
    for name, params in head.items():
        print(name, params.size(), params.dtype)
#print(model['word_embeddings_for_head']['weight']==embeddings['word_embeddings']['weight'])