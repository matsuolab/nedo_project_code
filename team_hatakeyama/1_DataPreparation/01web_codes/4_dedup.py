from concurrent.futures import ThreadPoolExecutor
import os
import glob

# integrate_web_datasetで処理されたカテゴリ化済みのjsonlをdedupする

import argparse

parser = argparse.ArgumentParser(
    description="Process files")
parser.add_argument('max_workers', type=str,
                    help='Number of parallel workers')
args = parser.parse_args()

# 並列プロセスの数
max_workers = int(args.max_workers)
print("Max workers: ", max_workers)


input_dir = "../data/categorized"
output_dir = "../data/dedup_categorized"


def make_dir(target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)


make_dir(output_dir)

# make dirs
dir_list = glob.glob(f"{input_dir}/*")
job_list = [i.split("/")[-1] for i in dir_list]
#for job_name in job_list:


def run_command(job_name):

    make_dir(f"{output_dir}/{job_name}")
    cmd = f"../dedup_sentence/deduplicate {input_dir}/{job_name}/ {output_dir}/{job_name}/"
    # cmd = f"../dedup/deduplicate {input_dir}/{job_name}/ {output_dir}/{job_name}/"
    os.system(cmd)


with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(run_command, job_list)

# %%
