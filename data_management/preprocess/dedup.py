import os
import json
import glob
import tqdm

path_jsonl = []
path_jsonl += sorted(glob.glob('./datasets/mC4/*.jsonl'))
path_jsonl += sorted(glob.glob('./datasets/OSCAR/*.jsonl'))
path_jsonl += sorted(glob.glob('./datasets/cc100/*.jsonl'))

path_output = './datasets/dedup_info.txt'

if os.path.exists(path_output):
    raise "file aleady exists."

seen = set()

with open('./datasets/dedup_info.txt', 'w') as f:
    for path in path_jsonl:
        _path = path.split('datasets/')[1]
        with open(path, 'r') as f_in:
            for i, line in tqdm.tqdm(enumerate(f_in), desc=_path):
                line = line.strip()
                if line == '':
                    continue
                data = json.loads(line)
                for lsh in data['meta']['dedup_LSHs']:
                    if lsh in seen:
                        info = f'{_path}\t{i}\n'
                        f.write(info)
                        break
                seen.update(data['meta']['dedup_LSHs'])

