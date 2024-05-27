# %%
import sys
if True:
    sys.path.append("../20integrate_texts/")
from tqdm import tqdm
import json
from loaders.RandomEnglishDataset import RandomEnglishDataset
ds = RandomEnglishDataset()

n_articles = 2*10**8

loader = iter(ds)

# %%
out_path = "/data/hatakeyama/python/eng_corpus/eng3.jsonl"
count = 0
with open(out_path, "w") as f:
    f.write("")

for i in tqdm(range(n_articles)):
    t = next(loader)["text"]
    if t == "":
        continue
    count += 1
    with open(out_path, "a") as f:
        f.write(json.dumps({"text": t}, ensure_ascii=False)+"\n")

with open(out_path.replace(".jsonl", "_count.txt"), "w") as f:
    f.write(str(count))
# %%
