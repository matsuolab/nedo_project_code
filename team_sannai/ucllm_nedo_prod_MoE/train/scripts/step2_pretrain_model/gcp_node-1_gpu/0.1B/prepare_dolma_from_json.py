from pathlib import Path
import json

path = Path('/persistentshare/storage/team_sannai/team_la/corpus/dolma_v1_6_sample')
filepaths = sorted(path.glob('*.json'))

for cnt, path in enumerate(filepaths):
    print(path)
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    grouped_datasets = {}
    for item in data:
        source = item['source']
        if source not in grouped_datasets:
            grouped_datasets[source] = []
        grouped_datasets[source].append(item)

    for source, items in grouped_datasets.items():    
        file_name = f"./dataset/{source}_{str(cnt).zfill(3)}.jsonl"
        with open(file_name, 'w') as file:
            for item in items:
                json_line = json.dumps(item)
                file.write(json_line + '\n')
        print(f"Saved {source} dataset to {file_name}")

