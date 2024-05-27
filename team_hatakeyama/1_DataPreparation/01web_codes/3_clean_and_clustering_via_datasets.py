# %%
# きれいなテキストを分配する
import os
# %%

import sys
if True:
    sys.path.append("../20integrate_texts/")
from loaders.loaders import *
from src.cleaner.auto_cleaner import clean_text
from src.classify.Text2Vec import Text2Vec, texts2classes
from src.cleaner.auto_cleaner import clean_text, ml_clean_text
import joblib
from src.distribute_jsonl import process_lines, make_dir
from gensim.models import KeyedVectors
from datasets import load_dataset
# %%
streaming = True
base_dir = "../data/categorized"
length_threshold = 30  # 短い記事は捨てる
check_length = 100  # はじめのlengthだけで分類する


# load models
# t2v = Text2Vec(load_facebook_model('../data/model/cc.ja.300.bin'))
t2v = Text2Vec(model=KeyedVectors.load_word2vec_format(
    '../data/model/entity_vector/entity_vector.model.bin', binary=True),
    dim=200,
)
kmeans = joblib.load("../data/model/kmeans.pkl")


# %%


def proc(docs, database_path):
    process_lines(docs, t2v, kmeans,
                  base_dir,
                  database_path,
                  check_length=check_length)


# %%
sys.path.append("../20integrate_texts/")
length_threshold = 100
batch_size = 100

loader_dict = {
     "kokkai": kokkai_loader(), #413
     "NHK_School": NHKSchool_loader(), #411済
     "WikiQA": wiki_qa_loader(),#411済
     "Wiki": cleaned_wiki_loader(),#411済
     "NHK_News": nhk_news_loader(),  # 411済
     "aozora": aozora_bunko_loader(),  # 411済
     "j_ronbun": j_research_loader(),  # 411済
     "cosmo": cosmo_loader(),  # 411済
     "novels": load_dataset("atsushi3110/novels-ja", split="train", streaming=streaming), #412
     "coding_blog": load_dataset("atsushi3110/coding-blog-ja", split="train", streaming=streaming), #412
     "en_ja_corpus_augumented_a": AltParallelEnJaDataset(repo_name="atsushi3110/en-ja-parallel-corpus-augmented",
                                                        data_files="https://huggingface.co/datasets/atsushi3110/en-ja-parallel-corpus-augmented/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
                                                        streaming=False),
     "en_ja_corpus_augumented_b": AltParallelEnJaDataset(repo_name="atsushi3110/en-ja-parallel-corpus-augmented",
                                                        data_files="https://huggingface.co/datasets/atsushi3110/en-ja-parallel-corpus-augmented/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
                                                        streaming=False),

     "ShosetsuSevenK": ShosetuSevenK(),  # 小説データ
     "SodaJa": SodaJaDataset(),  # sodaの日本語訳 413. エラーが出て止まるので注意
     "JaNews1": JaNewsDataset(data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news_aa.jsonl"),#414
     "JaNews2": JaNewsDataset(data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news_bb.jsonl"),#414
     "JaNews3": JaNewsDataset(data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news_cc.jsonl"),#414
     "JaNews4": JaNewsDataset(data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news.dd.jsonl"),#414
     "JaNews5": JaNewsDataset(data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news.ee.jsonl"), #textがないので割愛
     "JaNews6": JaNewsDataset(data_files="https://huggingface.co/datasets/atsushi3110/news-ja/resolve/main/news.ff.jsonl"), #414
     "light_novel_4m":LightNovelFourM(auth_token=os.environ["hf_key"],streaming=True), #ライトノベル。承認とapi keyが必要
    "JetCopper": load_dataset("sudy-super/JetCopper-10B", split="train", use_auth_token=os.environ["hf_key"], streaming=True,),
}

# %%

no_cleaning_list = [
    "en_ja_corpus_augumented_a",
    "en_ja_corpus_augumented_b",
    "ShosetsuSevenK",
    "SodaJa",
    "ShosetsuSevenK",
    "light_novel_4m",
]

for doc_name, loader in loader_dict.items():
    print(doc_name)
    docs = []
    lines = []
    for record in iter(loader):
        lines.append(record["text"])

    cnt = 0
    for text in lines:
        if doc_name not in no_cleaning_list:
            text = clean_text(text)
            if len(text) < length_threshold:
                continue

        docs.append(text)
        if len(docs) == batch_size:
            # docsのコピーを作成してprocに渡す
            proc(docs[:], doc_name+str(cnt))
            # docsをリセット
            docs = []
            # cnt+=1

    proc(docs[:], doc_name+str(cnt))
