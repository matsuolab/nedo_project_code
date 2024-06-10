from loaders import *
import argparse
from record_distributor import RecordDistributor
from record_merge import RecordMerger
from loaders import *
import json
import os
from tqdm import tqdm
import yaml
import os
import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT_BASE = os.path.join(ROOT_PATH, "output")

MAX_RECORDS = 20
OVERWRITE = True


def integrate_data(output_file="merge_text.jsonl", type="llama", batch_size=10):
    # 種々のdatasetをintagrate
    if not os.path.exists(OUTPUT_BASE):
        os.mkdir(OUTPUT_BASE)
 
    """
    loader: datasetsのloaderを入れてます｡ 
    n_records: 最大の学習件数
    stage_ratio: 各ステージで､どの程度､データを食わせるかを決めます｡

    例:
    wiki(ja): [1,1,9]
    wiki(en): [1,9,1]
    の場（）
    データセットを3stageに分けます｡

    1st stageでは､ wiki(ja)の10%､wiki(en)の10%を混ぜて学習させます
    2nd stageでは､ wiki(ja)の10%､wiki(en)の90%を混ぜて学習させます
    3rd stageでは､ wiki(ja)の90%､wiki(en)の10%を混ぜて学習させます

    このようなステージ分けをすることで､一種のカリキュラム学習を行うことが出来ます
    """
    stage_n_records = [MAX_RECORDS, MAX_RECORDS/2]

    dataset_dict = {
        "wiki(ja)": {
            "loader": test_wiki_ja_loader, #日本語版のwikipediaのloaderを使います｡
            #"n_records": MAX_RECORDS, #最大件数（type: mergeではstage_n_recordsを使用）
            "stage_ratio": [2, 3], # 各ステージでのデータ配分
        },

        "mc4(ja)": {
            "loader": test_mc4_ja_loader,
            #"n_records": MAX_RECORDS, #最大件数（type: llamaではstage_n_recordsを使用）
            "stage_ratio": [2, 2],
        },
    }
    if type == "distribute":
        distributor = RecordDistributor(dataset_dict)
        distributor.load_datasets()
        distributor.write_jsonl(os.path.join(OUTPUT_BASE, output_file), overwrite=OVERWRITE)

    elif type == "merge":
        # https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/A11-5.pdf
        distributor = RecordMerger(dataset_dict, stage_n_records, batch_size=batch_size)
        distributor.load_datasets()
        distributor.write_jsonl(os.path.join(OUTPUT_BASE, output_file), overwrite=OVERWRITE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process merge texts.')
    parser.add_argument('--output_file', type=str,
                        help='The output file name', required=False, default="merge_text.jsonl.gz")
    parser.add_argument('--batch_size', type=int,
                        help='set batch size', required=False, default=10)
    parser.add_argument('--type', type=str,
                        help='merge type', required=False, default="merge")
    args = parser.parse_args()
    output_file = args.output_file
    batch_size = args.batch_size
    merge_type = args.type

    integrate_data(output_file=output_file, type=merge_type, batch_size=batch_size)
