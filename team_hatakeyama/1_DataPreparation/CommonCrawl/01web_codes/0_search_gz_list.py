# %%
# gzファイルの一覧を取得
import glob
import os

gz_list = glob.glob("../data/original_dump/**/*.jsonl.gz", recursive=True)

mc4_gz_list = []
for path in glob.glob("../data/original_dump/c4/multilingual/*.json.gz"):
    s = os.path.getsize(path)
    if s > 1000:
        mc4_gz_list.append(path)

gz_list += mc4_gz_list

# 別途取得したparquetファイルの一覧を取得
# TODO:このディレクトリはhardcodeされているので、変更がある場合は修正が必要
pq_list = glob.glob(
    "/data/hatakeyama/python/llm_corpus_original/**/*.parquet", recursive=True)
pq_list = [i for i in pq_list if os.path.getsize(i) > 5000]
gz_list += pq_list


gz_list = list(set(gz_list))

# %%
print(len(gz_list), " files available")

# %%
with open("temp/gz_list.txt", "w") as f:
    for path in gz_list:
        f.write(path+"\n")

# %%
