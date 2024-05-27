# 各クラスタの
# 行数を数える
import json
from tqdm import tqdm
from loaders.CommonCrawlDataset import CommonCrawlDataset

with open("../data/clustered_path.json") as f:
    label_to_path_list = json.load(f)


label_to_article_count = {}
count = 0
for label, path_list in tqdm(label_to_path_list.items()):
    loader = CommonCrawlDataset(label_to_path_list[label])
    loader = (iter(loader))
    while (True):
        try:
            next(loader)
            count += 1

            if count % 100000 == 0:
                print(count)
        except:
            records = loader.count
            label_to_article_count[label] = records
            break


with open("../data/clustered_n.json", "w") as f:
    json.dump(label_to_article_count, f, indent=4)
