
import json
import os
from tqdm import tqdm
import yaml

with open('config.yaml', 'r') as file:
    conf= yaml.safe_load(file)
print(conf)


import sentencepiece as spm
model_path=conf["input_tokenizer_file"]#+"/tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=model_path)


#wikipedia 200万文章で20minほど
total_tokens=0
import json
from tqdm import tqdm
with open(conf["input_jsonl"],"r") as f:
    for line in tqdm(f):
        text=json.loads(line)["text"]
        n_tokens=len(sp.encode(text, out_type=str))
        total_tokens+=n_tokens



#billion
print("tokens in billion")
print(total_tokens/10**9)
print("tokens")
print(total_tokens)