from datasets import load_dataset

dataset_name = "izumi-lab/wikinews-en-20230728"
dataset = load_dataset(dataset_name)
dataset['train'].to_json("./train.jsonl", orient="records", lines=True,  force_ascii=False)