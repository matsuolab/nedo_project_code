from datasets import disable_caching, load_dataset, concatenate_datasets
import logging
import yaml

with open("/home//ucllm_nedo_dev/train/scripts/step4_finetune_model/config/config.yaml") as file:
    data_files = yaml.safe_load(file)['datasets']

for data_file in data_files:
    dataset = load_dataset(data_file['name'], split=data_file['split'])

for input_text in dataset['input']:
    print(input_text is None)
    break

# for data_file in data_files:
#     #dataset = load_dataset("json", data_files=data_file)
#     dataset = load_dataset(data_file['name'], split=data_file['split'])
#     #dataset = dataset[data_file['split']]
    
#     def create_propmt(example): 
#         prompt = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{example[data_file['instruction']]}"
#         #if data_file['input'] is not None:
#         if data_file['input'] in example and example[data_file['input']] is not None: 
#             prompt += f"\n\n### 入力:\n{example[data_file['input']]}"
#         prompt += f"\n\n### 応答:\n{example[data_file['output']]}"
#         return prompt

#     dataset = dataset.map(lambda example: {"text": create_propmt(example)})
#     dataset = dataset.select_columns("text")
#     logger.info(f"{data_file['name']}: {len(dataset)} records")
#     logger.info(f"example: {dataset[:5]}")