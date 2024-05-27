from loaders.loaders import *


# tokenizerの学習用｡日本語を多めにする｡
# from dataset_dict_tokenizer import dataset_dict, output_path

# 本番用
from dataset_dict import dataset_dict, output_path
from loaders.RecordDistributor import RecordDistributor


distributor = RecordDistributor(dataset_dict)
distributor.load_datasets()
print("begin writing jsonl")
distributor.write_jsonl(output_path, overwrite=True)
