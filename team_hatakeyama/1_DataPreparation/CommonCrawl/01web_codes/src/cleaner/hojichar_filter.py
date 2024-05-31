
from hojichar import Compose, document_filters
import json
import random

base_path = "src/cleaner/hoji_dict/"
cleaner = Compose([
    document_filters.JSONLoader(key="text"),
    document_filters.AcceptJapanese(),
    document_filters.DocumentLengthFilter(min_doc_len=10, max_doc_len=50000),
    document_filters.MaskPersonalInformation(),
    document_filters.JSONDumper(),
])

with open(base_path + "adult_keywords_ja.txt") as f:
    adult_keywords_ja = f.read().splitlines()
with open(base_path + "adult_keywords_en.txt") as f:
    adult_keywords_en = f.read().splitlines()
with open(base_path + "advertisement_keywords_ja.txt") as f:
    advertisement_keywords_ja = f.read().splitlines()

noise_keywords = adult_keywords_ja + adult_keywords_en + advertisement_keywords_ja
noise_keywords = list(set(noise_keywords))
noise_keywords = [k for k in noise_keywords if k != ""]

prob_cleaner = Compose([
    document_filters.JSONLoader(key="text"),
    # document_filters.DiscardRareKuten(),  # 日本語以外を消す
    document_filters.DiscardAdultContentJa(
        base_path + "adult_keywords_ja.txt"),
    document_filters.DiscardAdultContentEn(
        base_path + "adult_keywords_en.txt"
    ),
    # document_filters.DiscardDiscriminationContentJa(
    #    base_path + "discrimination_keywords_ja.txt"
    # ),
    # document_filters.DiscardViolenceContentJa(
    #    base_path + "violence_keywords_ja.txt"
    # ),
    document_filters.DiscardBBSComments(),
    document_filters.DiscardAds(
        base_path + "advertisement_keywords_ja.txt",
        max_allowed_num=10,
    ),
    document_filters.JSONDumper(),
])


def hoji_filter(text):
    d = {"text": text}
    parsed = cleaner(json.dumps(d))
    if parsed == "":
        return ""
    text = json.loads(parsed)["text"]
    return text


# 確率的にフィルタリングするが､ちょっとイマイチ
def prob_hoji_filter(text, survive_ratio=0.5):
    # ngワード類については､確率的に処理する
    if random.random() < survive_ratio:
        return text

    d = {"text": text}
    parsed = prob_cleaner(json.dumps(d))
    if parsed == "":
        return ""
    text = json.loads(parsed)["text"]
    return text


def prob_filter(text, ratio=5):
    # 遅い
    if not text:
        return ""

    # テキスト全体の長さ
    total_length = len(text)

    # noise_keywordsの各単語がテキストに含まれる回数の合計を計算
    noise_length = sum(text.count(word) for word in noise_keywords)

    # noise_keywordsがテキストに占める割合を計算
    noise_ratio = noise_length / total_length

    # print(noise_ratio*ratio)
    # 割合が閾値以上であれば、確率的に""を返す
    if noise_ratio*ratio > random.random():
        return ""
    else:
        return text
