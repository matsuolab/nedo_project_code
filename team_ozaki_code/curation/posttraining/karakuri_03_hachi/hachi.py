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

dataset = load_dataset("HachiML/Hachi-Alpaca-Mixtral-8x22B-Instruct-v0.1", split="v1.0_cleaned")

json_file_path = os.path.join('/home/user', 'hachi.jsonl')

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

for i, record in enumerate(dataset):
    no = record.get('No.')
    #if no < 26832:
    #    continue
    instruction_text = record.get('instruction')
    input_text = record.get('input')
    output_text = record.get('output')
    eos = False
    if not input_text:  # inputが空の場合
        prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示
{instruction_text}

### 応答
"""
    else:
        prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示
{instruction_text}

### 入力
{input_text}

### 応答
"""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attention_mask = input_ids.ne(tokenizer.pad_token_id).int()
    generated_text, elapsed_time = generate_response(model, tokenizer, input_ids, attention_mask, 512)
    if '</s>' not in generated_text:
        if not input_text:  # inputが空の場合
            prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を400字以内で書きなさい。

### 指示
{instruction_text}

### 応答
"""
        else:
            prompt = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を400字以内で書きなさい。

### 指示
{instruction_text}

### 入力
{input_text}

### 応答
"""
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        attention_mask = input_ids.ne(tokenizer.pad_token_id).int()
        generated_text, elapsed_time = generate_response(model, tokenizer, input_ids, attention_mask, 512)
    if '</s>' in generated_text:
        eos = True
    generated_text = generated_text.replace('</s>', '')
    data = {
        "No.": no,
        "prompt": prompt,
        "instruction": instruction_text,
        "input": input_text,
        "output": generated_text,
        "old_output": output_text,
        "eos": eos,
        'time': f"{elapsed_time:.2f}"
    }
    with open(json_file_path, 'a', encoding='utf-8') as f:
        # json.dump(data, f, ensure_ascii=False, indent=4)
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print("finished")


