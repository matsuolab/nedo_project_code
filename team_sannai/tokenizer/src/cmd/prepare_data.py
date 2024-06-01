from datasets import load_dataset
import os

def _is_real_folder(path):
    if os.path.isdir(path):
        return True
    else:
        return False

def prepare_data(input: str, key: str, category: str):
    if _is_real_folder(input):
        dataset = load_dataset('json', data_dir=input, split="train")
    else:    
        dataset = load_dataset(input, split="train")

    output_file = f"./artifacts/prepared/{category}.txt"
    with open(output_file, "w") as f:
        for line in dataset:
            text = line[key]
            f.write(text)
    print("output: ", output_file)
