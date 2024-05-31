# article数を愚直にカウントするscript
import glob
from tqdm import tqdm
jsonl_dir = "../data/dedup_categorized/**/*.jsonl"
web_jsonl_files = glob.glob(jsonl_dir, recursive=True)
print(len(web_jsonl_files), "files")

count = 0
for path in tqdm(web_jsonl_files):
    with open(path, "r") as f:
        lines = f.readlines()
        count += len(lines)
print(count, "articles")
with open("temp/count_articles.txt", "w") as f:
    f.write(str(count)+"articles \n")
    f.write(str(len(web_jsonl_files))+"files \n")
