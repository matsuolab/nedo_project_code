import os, json, datetime
import pandas as pd
from datetime import timezone, timedelta
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

# データセットのロード
dataset = load_dataset("GENIAC-Team-Ozaki/tuninig-dataset_pref_20pct", split="train")
df = pd.DataFrame(dataset)

model_path = "GENIAC-Team-Ozaki/lora-sft-finetuned-stage4-iter86000"
temperature = 1e-2
top_p = 1.0
max_tokens = 512

json_file_path = os.path.join('/home/ext_someya_ai_iisec_ac_jp', 'pref_20pct.jsonl')
skip_file_path = os.path.join('/home/ext_someya_ai_iisec_ac_jp', 'pref_20pct_skip.txt')
progress_file_path = os.path.join('/home/ext_someya_ai_iisec_ac_jp', 'progress.txt')

llm = LLM(
    model=model_path,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=False,
    use_fast=False,
)

def load_last_processed_index(progress_file_path):
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r', encoding='utf-8') as progress_file:
            last_index = progress_file.read().strip()
            return int(last_index) if last_index.isdigit() else -1
    return -1

def save_last_processed_index(progress_file_path, last_index):
    with open(progress_file_path, 'w', encoding='utf-8') as progress_file:
        progress_file.write(str(last_index))

def generate_responses(model, prompts, sampling_params):
    responses = model.generate(prompts, sampling_params)
    return responses

def process_batches(dataframe, batch_size=50):
    total_rows = len(dataframe)
    last_processed_index = load_last_processed_index(progress_file_path)

    start_index = last_processed_index + 1
    if start_index >= total_rows:
        print("All rows are already processed.")
        return

    processed_count = start_index

    while processed_count < total_rows:
        batch_end_index = min(processed_count + batch_size, total_rows)
        batch_df = dataframe.iloc[processed_count:batch_end_index]
        prompts = []
        indices = []

        # プロンプト作成
        for index, row in batch_df.iterrows():
            instruction = row['instruction']
            if row['input'] is not None and row['input'] != '':
                prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示
{instruction}

### 入力
{row['input']}

### 応答
"""
            else:
                prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示
{instruction}

### 応答
"""
            prompts.append(prompt)
            indices.append(index)

        try:
            generated_responses = generate_responses(llm, prompts, sampling_params)
            # バッチ内でイテレーションして応答を処理
            for local_idx, response, prompt in zip(indices, generated_responses, prompts):
                if response.outputs and len(response.outputs) > 0:
                    generated_text = response.outputs[0].text.strip()
                    data = {
                        "instruction": dataframe.at[local_idx, 'instruction'],
                        "input": dataframe.at[local_idx, 'input'],
                        "chosen": dataframe.at[local_idx, 'chosen'],
                        "rejected": generated_text,
                        "prompt": prompt,
                    }
                    with open(json_file_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                else:
                    print(f"No outputs for prompt: {prompt}")
                    with open(skip_file_path, 'a', encoding='utf-8') as skip_file:
                        skip_file.write(f"{local_idx}\n")
        except Exception as e:
            print(f"Error: {e}")
            with open(skip_file_path, 'a', encoding='utf-8') as skip_file:
                skip_file.write(f"{','.join(map(str, indices))}\n")

        # バッチごとにプロンプトとインデックスを初期化
        prompts = []
        indices = []

        # 最後に処理したインデックスを保存
        last_index = batch_end_index - 1
        save_last_processed_index(progress_file_path, last_index)

        processed_count = batch_end_index

process_batches(df, batch_size=50)

print("Finished processing.")