
import os, datetime, time, json
import pandas as pd
import torch
from datetime import timezone, timedelta
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

def generate_response(model, tokenizer, input_ids, attention_mask, max_new_tokens):
    start_time = time.time()
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.99, top_p=0.95)
    end_time = time.time()
    elapsed_time = end_time - start_time
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    return generated_text, elapsed_time

dataset = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
df = pd.DataFrame(dataset)

json_file_path = os.path.join('/home/user', 'dolly.jsonl')
skip_file_path = os.path.join('/home/user', 'dolly_skip.txt')

model = AutoAWQForCausalLM.from_quantized(
    "GENIAC-Team-Ozaki/karakuri-lm-8x7b-chat-v0.1-awq",
    fuse_layers=True,
    trust_remote_code=False,
    safetensors=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "GENIAC-Team-Ozaki/karakuri-lm-8x7b-chat-v0.1-awq",
    trust_remote_code=False,
)

# for i, record in enumerate(dataset):
for i, row in df.iterrows():
    index = row['index']
    #if int(index) < 4411:
    #    continue
    category = row['category']
    instruction = row['instruction']
    input = row['input']
    output = row['output']
    if category in ['open_qa', 'general_qa', 'classification', 'closed_qa']:
        if input == '':
            prompt = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。
ステップバイステップで考え、論理的な説明をもとに答えを導いてください。回答例を見たことについては言及しないでください。

### 指示
{instruction}

### 回答例
{output}

### 応答
"""
        else:
            prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。
ステップバイステップで考え、論理的な説明をもとに答えを導いてください。回答例を見たことについては言及しないでください。

### 指示
{instruction}

### 入力
{input}

### 回答例
{output}

### 応答
"""
    else: #QA以外
        if input == '':
            prompt = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示
{instruction}

### 応答
"""
        else:
            prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示
{instruction}

### 入力
{input}

### 応答
"""
    eos = False
    generated_text = ''
    try:
        if len(prompt)<=2000:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            attention_mask = input_ids.ne(tokenizer.pad_token_id).int()
            generated_text, elapsed_time = generate_response(model, tokenizer, input_ids, attention_mask, 1024)
            if '</s>' not in generated_text:
                generated_text, elapsed_time = generate_response(model, tokenizer, input_ids, attention_mask, 512*3)
            if '</s>' in generated_text:
                eos = True
            generated_text = generated_text.replace('</s>', '')
    except Exception as e:
        print(f"Error at index {index}: {e}")
        with open(skip_file_path, 'a', encoding='utf-8') as skip_file:
            skip_file.write(f"{index}\n")
        continue  # 例外が発生した場合はこのインデックスをスキップ
        
    data = {
        "index": index,
        "instruction": instruction,
        "input": input,
        "output": generated_text,
        "prompt": prompt,
        "old_output": output,
        "eos": eos,
        'time': f"{elapsed_time:.2f}"
    }
    with open(json_file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print("finished")

