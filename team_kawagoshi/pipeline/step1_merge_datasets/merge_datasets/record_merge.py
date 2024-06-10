import os
import numpy as np
import random
import json
from tqdm import tqdm
import gzip

class RecordMerger:
    def __init__(self, dataset_dict, stage_n_records, batch_size=1) -> None:
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.stage_n_records = stage_n_records
        self.n_stages = len(stage_n_records)
        self.update_records_per_stage(self.dataset_dict, self.stage_n_records)


    def load_datasets(self):
        for name, dataset_info in self.dataset_dict.items():
            print(f"loading {name}")
            dataset_info["dataset"] = dataset_info["loader"]()
        self.init_iterators()


    def init_iterators(self):
        for name, dataset_info in self.dataset_dict.items():
            dataset_info["dataset_iterator"] = iter(dataset_info["dataset"])


    def update_records_per_stage(self, dataset_dict, stage_n_records):
        total_stage_ratio_list = []
        for i in range(len(stage_n_records)):
            total_stage_ratio = 0
            for tmp_dataset_info in dataset_dict.values():
                total_stage_ratio += tmp_dataset_info["stage_ratio"][i]
            total_stage_ratio_list.append(total_stage_ratio)

        total_stage_ratio_list = np.array(total_stage_ratio_list)

        max_stage_records = np.array(np.zeros(len(stage_n_records)))
        for dataset_info in dataset_dict.values():
            dataset_info["stage_ratio"] = np.array(dataset_info["stage_ratio"])
            dataset_info["stage_ratio"] = dataset_info["stage_ratio"] /  total_stage_ratio_list
            stage_records = (np.array(dataset_info["stage_ratio"])*np.array(stage_n_records)).astype(int)
            max_stage_records = np.maximum.reduce([max_stage_records, stage_records])
            dataset_info["records_per_stage"] = list(stage_records)
        
        self.max_data_records_per_stage = max_stage_records


    def write_jsonl(self, output_path, overwrite=True):
        if overwrite:
            with gzip.open(output_path, "wt") as f:
                f.write("")
        else:
            if os.path.exists(output_path):
                print("file already exists")
                raise FileExistsError
        # write files
        for stage in range(self.n_stages):
            print(f"writing stage {stage}")
            text_list = []
            text_set_cnt = 0  
            for cnt in tqdm(range(int(self.max_data_records_per_stage[stage]))):
                batch_cnt = text_set_cnt % self.batch_size

                # 各データセットの出現頻度に応じて、データを吐き出していく
                for dataset_info in self.dataset_dict.values():
                    frequency = dataset_info["stage_ratio"][stage]
                    # frequency
                    if frequency*self.batch_size > batch_cnt:
                        text = next(dataset_info["dataset_iterator"])
                        text_list.append(text["text"])

                text_set_cnt += 1
                # バッチにデータが溜まったら、シャッフルして書き出す
                if len(text_list) % self.batch_size == 0 and len(text_list) != 0:
                    random.shuffle(text_list)

                    with gzip.open(output_path, "at") as f:
                        for text in text_list:
                            out_text = json.dumps({"text": text}, ensure_ascii=False)
                            f.write(out_text+"\n")
                    text_list = []
                    text_set_cnt = 0
