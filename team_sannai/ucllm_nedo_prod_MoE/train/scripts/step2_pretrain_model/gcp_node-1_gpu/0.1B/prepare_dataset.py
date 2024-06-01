from datasets import load_dataset
import json

dataset = load_dataset("allenai/dolma", split='train')
grouped_datasets = {}
for item in dataset:
    source = item['source']
    if source not in grouped_datasets:
        grouped_datasets[source] = []
    grouped_datasets[source].append(item)

for source, items in grouped_datasets.items():    
    file_name = f"./dataset/{source}_dataset.jsonl"
    with open(file_name, 'w') as file:
        for item in items:
            json_line = json.dumps(item)
            file.write(json_line + '\n')
    print(f"Saved {source} dataset to {file_name}")