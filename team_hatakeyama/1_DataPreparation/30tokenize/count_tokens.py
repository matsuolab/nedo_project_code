
import sentencepiece as spm
import json
import os
from tqdm import tqdm
import yaml

with open('config.yaml', 'r') as file:
    conf = yaml.safe_load(file)
print(conf)


model_path = conf["input_tokenizer_file"]  # +"/tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=model_path)


# wikipedia 200万文章で20minほど
total_tokens = 0
total_length=0
count=0
with open(conf["input"], "r") as f:
    for line in tqdm(f):
        text = json.loads(line)["text"]
        n_tokens = len(sp.encode(text, out_type=str))
        total_tokens += n_tokens

        total_length+=len(text)
        count+=1

# billion
print("tokens in billion")
print(total_tokens/10**9)
print("tokens")
print(total_tokens)
print("length")
print(total_length)
print("documents")
print(count)