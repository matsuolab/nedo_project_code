from loaders.loaders import *
import json
# 出力パス

scale = 1  # 練習時はscaleを小さくする
scale = 1.05  # データ欠損などがあるせいか､微妙に誤差があるので､少し小さめにする
output_path = f"/data/hatakeyama/python/llm_corpus/BTM_J2_corpus_scale_{scale}.jsonl"


# 自動でクラスタリングされたコーパス群の読み込み
n_clusters = 5
# クラスタリングされたweb系テキストのデータセット
with open("../data/clustered_path.json", "r") as f:
    label_to_path_list = json.load(f)

# データ数
with open("../data/clustered_n.json", "r") as f:
    label_to_article_count = json.load(f)
print(label_to_article_count)
cc_loader_dict = {}
for label, path in label_to_path_list.items():
    loader = CommonCrawlDataset(label_to_path_list[label])
    cc_loader_dict[label] = loader


dataset_dict = {
    # ----------------------------
    # stage 1

    # 英語
    "en": {
        # "loader":  CommonCrawlDataset(["/data/hatakeyama/python/eng_corpus/eng3.jsonl"]),
        "loader": load_dataset("json", split="train",
                               data_files="/data/hatakeyama/python/eng_corpus/eng3.jsonl",
                               streaming=True),
        "n_records": int(67500000/scale/100),
        "stage_ratio": [0.5, 1, 1, 1, 1, 1, 1],
    },

    # "pmc": {
    #    "loader": PMCDataset2,
    #    "n_records": int(2000000/scale),  # 値は適当
    #    "stage_ratio": [0.05, 700, 0.05, 0.05, 0.05, 0.05, 0.05],

    # },
    # ----------------------------
    # stage 2以降
    # 日本語の雑多な文章
    #
    "ja0": {
        "loader": cc_loader_dict["4"],
        "n_records": int(label_to_article_count["4"]/scale-10000),
        "stage_ratio": [0.05, 0.05, 0.05, 0.05, 1, 0.05, 0.05],
    },
    "ja1": {
        "loader": cc_loader_dict["1"],
        "n_records": int(label_to_article_count["1"]/scale-10000),

        "stage_ratio": [0.05, 0.05, 0.05, 1, 0.05, 0.05, 0.05],
    },
    "ja2": {
        "loader": cc_loader_dict["2"],
        "n_records": int(label_to_article_count["2"]/scale-10000),
        "stage_ratio": [0.05, 0.05, 1, 0.05, 0.05, 0.05, 0.05],
    },
    "ja3": {
        "loader": cc_loader_dict["3"],
        "n_records": int(label_to_article_count["3"]/scale-10000),
        "stage_ratio": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1],
    },
    "ja4": {
        "loader": cc_loader_dict["0"],
        "n_records": int(label_to_article_count["0"]/scale-10000),
        "stage_ratio": [0.05, 0.05, 0.05, 0.05, 0.05, 1, 0.05],
    },





}
